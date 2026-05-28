#!/usr/bin/env python3
"""Start LiteLLM proxy with proper env vars."""
import os
import sys

# Load just the OpenRouter key from .hermes/.env
env_path = os.path.expanduser("~/.hermes/.env")
with open(env_path) as f:
    for line in f:
        line = line.strip()
        if line.startswith("OPENROUTER_API_KEY"):
            key, val = line.split("=", 1)
            os.environ[key] = val
            break

os.environ["LITELLM_MASTER_KEY"] = "sk-loc...1234"

os.chdir("/Users/atliqmini1/Desktop/litellm")
sys.argv = ["litellm", "--config", "config.local.yaml", "--port", "4000"]

from litellm.proxy.proxy_cli import run_server
run_server()
