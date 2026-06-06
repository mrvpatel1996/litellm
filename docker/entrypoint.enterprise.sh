#!/bin/bash
# ── LiteLLM Enterprise Docker Entrypoint ──
# Starts proxy with --use_prisma_db_push to auto-create tables.
set -e

echo "── LiteLLM Enterprise Startup ──"
echo "Starting proxy with auto DB migration..."

exec litellm --config /app/config.local.yaml --port 4000 --use_prisma_db_push
