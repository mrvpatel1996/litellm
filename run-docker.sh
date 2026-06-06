#!/bin/bash
set -e

# ── LiteLLM Enterprise ── Quick Docker Run ──
# Runs the litellm-enterprise image as a standalone container.
# For full stack, use: docker compose -f docker-compose.enterprise.yml up -d

cd ~/Desktop/litellm

# Load .env
set -a; source .env; set +a

PORT="${1:-4001}"
echo "Starting LiteLLM Enterprise on port $PORT..."

docker rm -f litellm-test 2>/dev/null || true

docker run -d --name litellm-test \
  -p ${PORT}:4000 \
  -e "DATABASE_URL=${DATABASE_URL}" \
  -e "LITELLM_MASTER_KEY=${LITELLM_MASTER_KEY}" \
  -e "OPENROUTER_API_KEY=${OPENROUTER_API_KEY}" \
  -e "UI_USERNAME=${UI_USERNAME:-admin}" \
  -e "UI_PASSWORD=${UI_PASSWORD:-admin}" \
  -e "QDRANT_VECTOR_SIZE=384" \
  -v $(pwd)/config.docker.yaml:/app/config.local.yaml:ro \
  litellm-enterprise:local

echo "Waiting 20s for startup..."
sleep 20

echo "=== Health check ==="
curl -sf http://localhost:${PORT}/health/liveliness && echo " ✅" || echo " ❌ Not ready yet"
echo ""
echo "Access: http://localhost:${PORT}/ui"
