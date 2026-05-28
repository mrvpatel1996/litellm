#!/bin/bash
source ~/.hermes/.env

echo "KEY=${OPENROUTER_API_KEY:0:12}..."

curl -s -m 15 -X POST http://localhost:4000/embeddings \
  -H 'Authorization: Bearer *** \
  -H 'Content-Type: application/json' \
  -d '{"model":"text-embedding-3-small","input":"test"}' \
  > /tmp/emb_test.json

cat /tmp/emb_test.json | head -c 300
echo ""
