#!/bin/bash
set -e
export OPENROUTER_API_KEY=$(python3 -c "
with open('/Users/atliqmini1/.hermes/.env') as f:
    for line in f:
        if line.startswith('OPENROUTER_API_KEY='):
            print(line.strip().split('=',1)[1])
            break
")
export LITELLM_MASTER_KEY=*** Bearer ***
cd /Users/atliqmini1/Desktop/litellm
exec .venv/bin/litellm --config config.local.yaml --port 4000
