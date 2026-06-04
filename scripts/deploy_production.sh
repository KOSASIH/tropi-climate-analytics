#!/bin/bash
# Production deployment script
# Requires: DEPLOY_KEY, PRODUCTION_HOST environment variables

set -euo pipefail

echo "[DEPLOY] Tropi-Climate-Analytics — Production Deployment"
echo "[DEPLOY] $(date -u '+%Y-%m-%dT%H:%M:%SZ')"

# Validate required environment
: "${PRODUCTION_HOST:?PRODUCTION_HOST not set}"
: "${DEPLOY_KEY:?DEPLOY_KEY not set}"

IMAGE_TAG="${GITHUB_SHA:-latest}"
REGISTRY="ghcr.io/kosasih/tropi-climate-analytics"

echo "[DEPLOY] Image: ${REGISTRY}:${IMAGE_TAG}"

# Setup SSH key
mkdir -p ~/.ssh
echo "${DEPLOY_KEY}" > ~/.ssh/deploy_key
chmod 600 ~/.ssh/deploy_key
ssh-keyscan -H "${PRODUCTION_HOST}" >> ~/.ssh/known_hosts 2>/dev/null

SSH="ssh -i ~/.ssh/deploy_key -o StrictHostKeyChecking=no ubuntu@${PRODUCTION_HOST}"

# Pull and restart on remote
$SSH << REMOTE_COMMANDS
set -euo pipefail
cd /opt/tropi-climate-analytics
docker pull ${REGISTRY}:${IMAGE_TAG}
sed -i "s|image: .*tropi-climate.*|image: ${REGISTRY}:${IMAGE_TAG}|g" docker-compose.prod.yml
docker-compose -f docker-compose.prod.yml up -d --no-deps --remove-orphans api
docker-compose -f docker-compose.prod.yml ps
REMOTE_COMMANDS

echo "[DEPLOY] Deployment complete: ${REGISTRY}:${IMAGE_TAG}"
