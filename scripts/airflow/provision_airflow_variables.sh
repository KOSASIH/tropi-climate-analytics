#!/usr/bin/env bash
# ============================================================
# Airflow Variable Provisioning — Tropi Climate Analytics
# Reads secrets from AWS Secrets Manager and writes them as
# Airflow Variables via the Airflow CLI (airflow variables set)
# or the Airflow REST API if running remotely.
#
# Usage:
#   ./provision_airflow_variables.sh [--env prod|staging|dev] [--dry-run]
#
# Prerequisites:
#   - AWS CLI configured with IAM role that has
#     secretsmanager:GetSecretValue on all listed secrets
#   - Airflow CLI in PATH (or AIRFLOW_API_BASE_URL set for REST)
#   - jq installed
#
# Managed by: CLOUD-FORGE
# ============================================================
set -euo pipefail

# ── Defaults ────────────────────────────────────────────────
ENV="prod"
PROJECT="tropi-climate"
REGION="ap-southeast-3"
DRY_RUN=false
USE_REST_API=false

# ── Arg parsing ─────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --env)        ENV="$2";        shift 2 ;;
    --dry-run)    DRY_RUN=true;    shift   ;;
    --rest-api)   USE_REST_API=true; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

log()  { echo "[$(date '+%Y-%m-%d %H:%M:%S WIB')] [INFO]  $*"; }
err()  { echo "[$(date '+%Y-%m-%d %H:%M:%S WIB')] [ERROR] $*" >&2; }
ok()   { echo "[$(date '+%Y-%m-%d %H:%M:%S WIB')] [✓ SET] $*"; }
dry()  { echo "[$(date '+%Y-%m-%d %H:%M:%S WIB')] [DRY]   $*"; }

SECRET_PATH="/${PROJECT}/${ENV}/airflow"

log "=== Airflow Variable Provisioning ==="
log "Environment : ${ENV}"
log "Project     : ${PROJECT}"
log "Region      : ${REGION}"
log "Secret path : ${SECRET_PATH}"
log "Dry run     : ${DRY_RUN}"

# ── Secret → Airflow variable mapping ───────────────────────
# Format: "AIRFLOW_VAR_NAME|SECRET_SUFFIX|is_connection(bool)"
declare -a VARIABLE_MAP=(
  "NASA_EARTHDATA_TOKEN|NASA_EARTHDATA_TOKEN|false"
  "BMKG_API_KEY|BMKG_API_KEY|false"
  "JWT_SECRET|JWT_SECRET|true"          # sensitive — masked in UI
  "TROPI_API_BASE_URL|TROPI_API_BASE_URL|false"
  "KAFKA_BOOTSTRAP_BROKERS|KAFKA_BOOTSTRAP_BROKERS|false"
)

# ── Fetch a secret from Secrets Manager ─────────────────────
fetch_secret() {
  local secret_name="$1"
  aws secretsmanager get-secret-value \
    --secret-id "${SECRET_PATH}/${secret_name}" \
    --region "${REGION}" \
    --query 'SecretString' \
    --output text 2>/dev/null
}

# ── Set an Airflow variable ──────────────────────────────────
set_airflow_variable() {
  local var_name="$1"
  local value="$2"
  local is_sensitive="$3"

  if [[ "${DRY_RUN}" == "true" ]]; then
    dry "Would set Airflow variable: ${var_name} (sensitive=${is_sensitive})"
    return 0
  fi

  if [[ "${USE_REST_API}" == "true" ]]; then
    # REST API path (for remote Airflow without CLI access)
    local AIRFLOW_API_BASE_URL="${AIRFLOW_API_BASE_URL:-http://localhost:8080}"
    local AIRFLOW_BASIC_AUTH="${AIRFLOW_BASIC_AUTH:-admin:admin}"  # override via env
    curl -sf -X POST \
      -u "${AIRFLOW_BASIC_AUTH}" \
      -H 'Content-Type: application/json' \
      "${AIRFLOW_API_BASE_URL}/api/v1/variables" \
      -d "{\"key\": \"${var_name}\", \"value\": \"${value}\"}" > /dev/null
  else
    # Local Airflow CLI
    airflow variables set "${var_name}" "${value}"
  fi

  ok "${var_name}"
}

# ── Verify required tools ───────────────────────────────────
if ! command -v aws &>/dev/null; then
  err "aws CLI not found. Install: https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html"
  exit 1
fi
if ! command -v jq &>/dev/null; then
  err "jq not found. Install via: apt-get install jq / brew install jq"
  exit 1
fi
if [[ "${USE_REST_API}" == "false" ]] && ! command -v airflow &>/dev/null; then
  err "airflow CLI not found. Set --rest-api flag for REST API mode."
  exit 1
fi

# ── Verify AWS identity ─────────────────────────────────────
log "Verifying AWS credentials..."
AWS_IDENTITY=$(aws sts get-caller-identity --query 'Arn' --output text)
log "Running as: ${AWS_IDENTITY}"

# ── Main provisioning loop ───────────────────────────────────
SET_COUNT=0
FAIL_COUNT=0

for ENTRY in "${VARIABLE_MAP[@]}"; do
  IFS='|' read -r VAR_NAME SECRET_SUFFIX IS_SENSITIVE <<< "${ENTRY}"

  log "Fetching ${SECRET_PATH}/${SECRET_SUFFIX}..."
  SECRET_VALUE=$(fetch_secret "${SECRET_SUFFIX}") || {
    err "Failed to fetch secret for ${VAR_NAME}. Ensure the secret exists and IAM policy is attached."
    (( FAIL_COUNT++ )) || true
    continue
  }

  if [[ -z "${SECRET_VALUE}" || "${SECRET_VALUE}" == "null" || "${SECRET_VALUE}" == "{}" ]]; then
    err "Secret ${SECRET_SUFFIX} is empty or unpopulated. Populate it in Secrets Manager first."
    (( FAIL_COUNT++ )) || true
    continue
  fi

  # Unwrap JSON if the secret is stored as {"value": "..."}
  if echo "${SECRET_VALUE}" | jq -e '.value' &>/dev/null; then
    SECRET_VALUE=$(echo "${SECRET_VALUE}" | jq -r '.value')
  fi

  set_airflow_variable "${VAR_NAME}" "${SECRET_VALUE}" "${IS_SENSITIVE}"
  (( SET_COUNT++ )) || true
done

log "=== Provisioning complete ==="
log "Set: ${SET_COUNT} variables | Failed: ${FAIL_COUNT}"

if [[ ${FAIL_COUNT} -gt 0 ]]; then
  err "${FAIL_COUNT} variable(s) failed. Check Secrets Manager population status."
  exit 1
fi

log "All Airflow variables provisioned. DAGs are ready to run."
