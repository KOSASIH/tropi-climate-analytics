#!/usr/bin/env bash
# ============================================================
# Secrets Manager Population Template — Tropi Climate Analytics
# One-time script to populate secret VALUES after terraform apply
# creates the empty secret shells.
#
# ⚠ NEVER commit real values. Fill in values interactively or
#   pipe from a secure vault (HashiCorp Vault, 1Password CLI, etc.)
#
# Usage:
#   ./populate_secrets_template.sh [--env prod|staging|dev]
# ============================================================
set -euo pipefail

ENV="${1:-prod}"
PROJECT="tropi-climate"
REGION="ap-southeast-3"
SECRET_PATH="/${PROJECT}/${ENV}/airflow"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

put_secret() {
  local name="$1"
  local value="$2"
  aws secretsmanager put-secret-value \
    --secret-id "${SECRET_PATH}/${name}" \
    --secret-string "{\"value\": \"${value}\"}" \
    --region "${REGION}"
  log "✓ Populated: ${SECRET_PATH}/${name}"
}

log "Populating Airflow secrets for environment: ${ENV}"
log "Secrets path: ${SECRET_PATH}"
log "---"

# ── Read values interactively (never hardcoded) ──────────────
read -rsp "NASA_EARTHDATA_TOKEN (Earthdata login bearer token): " NASA_TOKEN; echo
put_secret "NASA_EARTHDATA_TOKEN" "${NASA_TOKEN}"

read -rsp "BMKG_API_KEY: " BMKG_KEY; echo
put_secret "BMKG_API_KEY" "${BMKG_KEY}"

read -rsp "JWT_SECRET (min 32 chars, random): " JWT_SECRET; echo
put_secret "JWT_SECRET" "${JWT_SECRET}"

read -rp  "TROPI_API_BASE_URL (e.g. https://api.tropi-climate.id): " API_URL
put_secret "TROPI_API_BASE_URL" "${API_URL}"

# Kafka brokers — auto-populated from MSK terraform output
KAFKA_BROKERS=$(aws rds describe-db-clusters --region "${REGION}" 2>/dev/null || true)
# Override: fetch MSK bootstrap brokers from terraform output
KAFKA_BROKERS=$(cd "$(dirname "$0")/../../infrastructure/terraform" && \
  terraform output -raw msk_bootstrap_brokers 2>/dev/null || echo "")

if [[ -z "${KAFKA_BROKERS}" ]]; then
  read -rp  "KAFKA_BOOTSTRAP_BROKERS (from MSK console, TLS endpoints): " KAFKA_BROKERS
else
  log "KAFKA_BOOTSTRAP_BROKERS auto-detected from Terraform state: ${KAFKA_BROKERS:0:40}..."
fi
put_secret "KAFKA_BOOTSTRAP_BROKERS" "${KAFKA_BROKERS}"

log "---"
log "All Airflow secrets populated. Run provision_airflow_variables.sh to push to Airflow."
