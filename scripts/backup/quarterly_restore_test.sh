#!/usr/bin/env bash
# ============================================================
# Quarterly Restore Test — Tropi Climate Analytics
# Tests RDS point-in-time restore to an isolated test cluster
# RTO target: 4 hours | RPO target: 1 hour
# Schedule: Quarterly (Jan/Apr/Jul/Oct 1st, 02:00 WIB)
# Evidence saved to S3 for compliance records
# Managed by: CLOUD-FORGE
# ============================================================
set -euo pipefail

DATE=$(date +%Y-%m-%d)
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
ENV=${ENVIRONMENT:-prod}
PROJECT="tropi-climate"
REGION="ap-southeast-3"
BACKUP_BUCKET="${PROJECT}-${ENV}-backups"
SOURCE_CLUSTER="${PROJECT}-${ENV}"
TEST_CLUSTER="${PROJECT}-dr-test-${TIMESTAMP}"
RTO_TARGET=14400   # 4 hours in seconds
RPO_TARGET=3600    # 1 hour in seconds

log()  { echo "[$(date '+%Y-%m-%d %H:%M:%S WIB')] [INFO]  $*" | tee -a /tmp/dr-test-${TIMESTAMP}.log; }
err()  { echo "[$(date '+%Y-%m-%d %H:%M:%S WIB')] [ERROR] $*" | tee -a /tmp/dr-test-${TIMESTAMP}.log >&2; }
pass() { echo "[$(date '+%Y-%m-%d %H:%M:%S WIB')] [✓ PASS] $*" | tee -a /tmp/dr-test-${TIMESTAMP}.log; }
fail() { echo "[$(date '+%Y-%m-%d %H:%M:%S WIB')] [✗ FAIL] $*" | tee -a /tmp/dr-test-${TIMESTAMP}.log; }

START_TIME=$(date +%s)
RESULT="PASS"

log "=== QUARTERLY DR RESTORE TEST — ${DATE} ==="
log "Source cluster: ${SOURCE_CLUSTER}"
log "Test cluster:   ${TEST_CLUSTER}"
log "RTO target: ${RTO_TARGET}s (4h) | RPO target: ${RPO_TARGET}s (1h)"

# ─── 1. Identify latest backup snapshot ─────────────────────────
log "Identifying latest RDS snapshot..."
LATEST_SNAPSHOT=$(aws rds describe-db-cluster-snapshots \
  --db-cluster-identifier "${SOURCE_CLUSTER}" \
  --region "${REGION}" \
  --query 'sort_by(DBClusterSnapshots, &SnapshotCreateTime)[-1].DBClusterSnapshotIdentifier' \
  --output text)
log "Latest snapshot: ${LATEST_SNAPSHOT}"

# Verify snapshot age is within RPO
SNAPSHOT_TIME=$(aws rds describe-db-cluster-snapshots \
  --db-cluster-snapshot-identifier "${LATEST_SNAPSHOT}" \
  --region "${REGION}" \
  --query 'DBClusterSnapshots[0].SnapshotCreateTime' \
  --output text)
SNAPSHOT_EPOCH=$(date -d "${SNAPSHOT_TIME}" +%s 2>/dev/null || date -j -f "%Y-%m-%dT%H:%M:%S" "${SNAPSHOT_TIME}" +%s)
CURRENT_EPOCH=$(date +%s)
SNAPSHOT_AGE=$((CURRENT_EPOCH - SNAPSHOT_EPOCH))

if [[ ${SNAPSHOT_AGE} -le ${RPO_TARGET} ]]; then
  pass "RPO check: snapshot age ${SNAPSHOT_AGE}s <= ${RPO_TARGET}s target"
else
  fail "RPO check: snapshot age ${SNAPSHOT_AGE}s > ${RPO_TARGET}s target"
  RESULT="FAIL"
fi

# ─── 2. Restore to isolated test cluster ───────────────────────
log "Initiating restore to test cluster ${TEST_CLUSTER}..."
RESTORE_START=$(date +%s)

aws rds restore-db-cluster-from-snapshot \
  --db-cluster-identifier "${TEST_CLUSTER}" \
  --snapshot-identifier "${LATEST_SNAPSHOT}" \
  --engine aurora-postgresql \
  --engine-version "15.4" \
  --region "${REGION}" \
  --db-cluster-parameter-group-name "${PROJECT}-${ENV}-pg15-params" \
  --deletion-protection false \
  --tags Key=Purpose,Value=dr-test Key=Date,Value="${DATE}"

# Add a reader instance to test cluster
aws rds create-db-instance \
  --db-instance-identifier "${TEST_CLUSTER}-instance" \
  --db-cluster-identifier "${TEST_CLUSTER}" \
  --db-instance-class "db.r6g.large" \
  --engine aurora-postgresql \
  --region "${REGION}"

log "Waiting for test cluster to become available (RTO target: ${RTO_TARGET}s)..."
aws rds wait db-cluster-available \
  --db-cluster-identifier "${TEST_CLUSTER}" \
  --region "${REGION}"

RESTORE_END=$(date +%s)
ACTUAL_RTO=$((RESTORE_END - RESTORE_START))

if [[ ${ACTUAL_RTO} -le ${RTO_TARGET} ]]; then
  pass "RTO check: restore completed in ${ACTUAL_RTO}s <= ${RTO_TARGET}s ($(( ACTUAL_RTO / 60 )) min)"
else
  fail "RTO check: restore took ${ACTUAL_RTO}s > ${RTO_TARGET}s target"
  RESULT="FAIL"
fi

# ─── 3. Validate data integrity ─────────────────────────────────
log "Validating restored data integrity..."
TEST_ENDPOINT=$(aws rds describe-db-clusters \
  --db-cluster-identifier "${TEST_CLUSTER}" \
  --region "${REGION}" \
  --query 'DBClusters[0].Endpoint' \
  --output text)

# Run basic validation queries via psql
if PGPASSWORD="${DB_PASSWORD}" psql -h "${TEST_ENDPOINT}" -U tropi_admin -d tropi_climate -c \
  "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='public';" 2>/dev/null; then
  pass "Data integrity: public schema tables accessible"
else
  fail "Data integrity: could not query restored database"
  RESULT="FAIL"
fi

# ─── 4. Cleanup test cluster ──────────────────────────────────
log "Cleaning up test cluster..."
aws rds delete-db-instance \
  --db-instance-identifier "${TEST_CLUSTER}-instance" \
  --skip-final-snapshot \
  --region "${REGION}" || true

aws rds wait db-instance-deleted \
  --db-instance-identifier "${TEST_CLUSTER}-instance" \
  --region "${REGION}" || true

aws rds delete-db-cluster \
  --db-cluster-identifier "${TEST_CLUSTER}" \
  --skip-final-snapshot \
  --region "${REGION}"
log "Test cluster deleted: ${TEST_CLUSTER}"

# ─── 5. Write evidence report to S3 ────────────────────────────
TOTAL_DURATION=$(( $(date +%s) - START_TIME ))

REPORT=$(cat <<EOF
{
  "test_date":        "${DATE}",
  "timestamp":        "${TIMESTAMP}",
  "environment":      "${ENV}",
  "result":           "${RESULT}",
  "source_cluster":   "${SOURCE_CLUSTER}",
  "test_cluster":     "${TEST_CLUSTER}",
  "snapshot_used":    "${LATEST_SNAPSHOT}",
  "snapshot_age_s":   ${SNAPSHOT_AGE},
  "rpo_target_s":     ${RPO_TARGET},
  "rpo_met":          $([ "${SNAPSHOT_AGE}" -le "${RPO_TARGET}" ] && echo true || echo false),
  "restore_duration_s": ${ACTUAL_RTO},
  "rto_target_s":     ${RTO_TARGET},
  "rto_met":          $([ "${ACTUAL_RTO}" -le "${RTO_TARGET}" ] && echo true || echo false),
  "total_test_duration_s": ${TOTAL_DURATION},
  "next_test_quarter":  "$(date -d '+3 months' +%Y-%m-01 2>/dev/null || date -v+3m +%Y-%m-01)"
}
EOF
)

echo "${REPORT}" | aws s3 cp - \
  "s3://${BACKUP_BUCKET}/dr-test-evidence/${DATE}/quarterly-dr-report.json" \
  --region "${REGION}" \
  --content-type application/json

aws s3 cp "/tmp/dr-test-${TIMESTAMP}.log" \
  "s3://${BACKUP_BUCKET}/dr-test-evidence/${DATE}/dr-test-${TIMESTAMP}.log" \
  --region "${REGION}"

log "Evidence report: s3://${BACKUP_BUCKET}/dr-test-evidence/${DATE}/quarterly-dr-report.json"
log "=== QUARTERLY DR TEST RESULT: ${RESULT} ==="

[[ "${RESULT}" == "PASS" ]] && exit 0 || exit 1
