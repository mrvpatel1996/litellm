#!/usr/bin/env python3
"""Write .env file for LiteLLM with proper credentials."""
import os

# Read OpenRouter key
or_key = ""
with open(os.path.expanduser("~/.hermes/.env")) as f:
    for line in f:
        line = line.strip()
        if line.startswith("OPENROUTER_API_KEY"):
            _, or_key = line.split("=", 1)
            break

lines = [
    "# LiteLLM Environment Variables",
    "# =================================",
    "",
    "# PostgreSQL Database",
    "DATABASE_URL=postgresql://litellm:***@localhost:5434/litellm",
    "",
    "# LiteLLM Master Key",
    "LITELLM_MASTER_KEY=sk-loc...1234",
    "",
    "# UI Login Credentials",
    "UI_USERNAME=admin",
    "UI_PASSWORD=*** OpenRouter API Key",
    "OPENROUTER_API_KEY=" + or_key,
]

with open("/Users/atliqmini1/Desktop/litellm/.env", "w") as f:
    f.write("\n".join(lines) + "\n")

print("Done! .env written")
