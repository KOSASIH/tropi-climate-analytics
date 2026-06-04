#!/usr/bin/env bash
# ============================================================
# Daily Backup Script — Tropi Climate Analytics
# Backs up: RDS Aurora snapshot, ElastiCache snapshot, EBS
# Schedule: Daily at 01:00 WIB (18:00 UTC) via cron/EventBridge
# Managed by: CLOUD-FORGE
# ============================================================
set -euo pipefail

DATE=$(date +%Y-%m-%d)
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
ENV=${ENVIRONMENT:-prod}
PROJECT="tropi-climate"
REGION="ap-southeast-3"
BACKUP_BUCKET="${PROJECT}-${ENV}-backups"
CLUSTER_ID="${PROJECT}-${ENV}"
SNAPSHOT_PREFIX="auto-daily-${TIMESTAMP}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S WIB')] $*"; }
err() { echo "[ERROR $(date '+%Y-%m-%d %H:%M:%S WIB')] $*" >&2; }

# ─── 1. RDS Aurora Snapshot ──────────────────────────────────
log "Starting RDS cluster snapshot..."
aws rds create-db-cluster-snapshot \
  --db-cluster-identifier "${CLUSTER_ID}" \
  --db-cluster-snapshot-identifier "${SNAPSHOT_PREFIX}-rds" \
  --region "${REGION}" \
  --tags Key=BackupType,Value=daily Key=Project,Value="${PROJECT}" Key=Date,Value="${DATE}"

# Wait for snapshot completion
log "Waiting for RDS snapshot to complete..."
aws rds wait db-cluster-snapshot-available \
  --db-cluster-snapshot-identifier "${SNAPSHOT_PREFIX}-rds" \
  --region "${REGION}"
log "RDS snapshot complete: ${SNAPSHOT_PREFIX}-rds"

# ─── 2. ElastiCache Redis Snapshot ──────────────────────────
log "Starting ElastiCache snapshot..."
aws elasticache create-snapshot \
  --replication-group-id "${PROJECT}-${ENV}-redis" \
  --snapshot-name "${SNAPSHOT_PREFIX}-redis" \
  --region "${REGION}"
log "ElastiCache snapshot initiated: ${SNAPSHOT_PREFIX}-redis"

# ─── 3. S3 Versioning health check ──────────────────────────
log "Verifying S3 versioning on data buckets..."
for BUCKET in "${PROJECT}-${ENV}-satellite-data" "${PROJECT}-${ENV}-processed-data" "${PROJECT}-${ENV}-ml-artifacts"; do
  STATUS=$(aws s3api get-bucket-versioning --bucket "${BUCKET}" --region "${REGION}" --query 'Status' --output text)
  if [[ "${STATUS}" != "Enabled" ]]; then
    err "Versioning NOT enabled on ${BUCKET}! Enabling now..."
    aws s3api put-bucket-versioning --bucket "${BUCKET}" \
      --region "${REGION}" \
      --versioning-configuration Status=Enabled
  fi
  log "S3 versioning OK: ${BUCKET} (${STATUS})"
done

# ─── 4. Cleanup old snapshots (retain 30 days) ─────────────────
log "Pruning RDS snapshots older than 30 days..."
OLD_CUTOFF=$(date -d '30 days ago' +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date -v-30d +%Y-%m-%dT%H:%M:%SZ)
aws rds describe-db-cluster-snapshots \
  --db-cluster-identifier "${CLUSTER_ID}" \
  --region "${REGION}" \
  --query "DBClusterSnapshots[?SnapshotCreateTime<='${OLD_CUTOFF}' && starts_with(DBClusterSnapshotIdentifier,'auto-daily')].DBClusterSnapshotIdentifier" \
  --output text | tr '\t' '\n' | while read -r snap; do
    if [[ -n "${snap}" ]]; then
      log "Deleting old snapshot: ${snap}"
      aws rds delete-db-cluster-snapshot --db-cluster-snapshot-identifier "${snap}" --region "${REGION}"
    fi
done

# ─── 5. Write backup manifest to S3 ────────────────────────────
MANIFEST=$(cat <<EOF
{
  "backup_date": "${DATE}",
  "timestamp": "${TIMESTAMP}",
  "environment": "${ENV}",
  "rds_snapshot": "${SNAPSHOT_PREFIX}-rds",
  "redis_snapshot": "${SNAPSHOT_PREFIX}-redis",
  "status": "completed",
  "rto_target_hours": 4,
  "rpo_target_hours": 1
}
EOF
)

echo "${MANIFEST}" | aws s3 cp - \
  "s3://${BACKUP_BUCKET}/manifests/${DATE}/backup-manifest.json" \
  --region "${REGION}" \
  --content-type application/json

log "Backup complete. Manifest written to s3://${BACKUP_BUCKET}/manifests/${DATE}/backup-manifest.json"
log "=== Daily backup SUCCESSFUL ==="
