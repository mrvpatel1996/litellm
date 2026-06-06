#!/usr/bin/env python3
"""Start LiteLLM proxy with .env from the litellm folder."""
import os
import sys
from pathlib import Path

LITELLM_DIR = Path("/Users/atliqmini1/Desktop/litellm")
env_file = LITELLM_DIR / ".env"

# Load .env from litellm folder
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                os.environ[k] = v

# Ensure vector size matches our bge-small model (384-dim)
# This is used by QdrantSemanticCache if qdrant_semantic_cache_vector_size
# is not explicitly passed (fallback default is 1536 for OpenAI ada).
os.environ.setdefault("QDRANT_VECTOR_SIZE", "384")

os.chdir(str(LITELLM_DIR))
sys.argv = ["litellm", "--config", "config.local.yaml", "--port", "4000"]

from litellm.proxy.proxy_cli import run_server
run_server()
