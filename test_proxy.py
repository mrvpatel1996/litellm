#!/usr/bin/env python3
"""Test LiteLLM proxy endpoints."""
import requests
import os

LT = "http://localhost:4000"
KEY = os.environ.get("LITELLM_MASTER_KEY", "sk-loc...1234")

# Load key from .env if not in env
if not os.environ.get("LITELLM_MASTER_KEY"):
    for line in open("/Users/atliqmini1/Desktop/litellm/.env"):
        if line.startswith("LITELLM_MASTER_KEY"):
            KEY = line.split("=", 1)[1].strip()
            break

auth = "Bearer " + KEY
hdrs = {"Authorization": auth}

print("1. Health check...")
r = requests.get(LT + "/health")
print("   Status:", r.status_code, "->", r.text[:80])

print("\n2. Auth with master key...")
r = requests.get(LT + "/health", headers=hdrs)
print("   Status:", r.status_code, "->", r.text[:80])

print("\n3. List models...")
r = requests.get(LT + "/v1/models", headers=hdrs)
print("   Status:", r.status_code)
if r.status_code == 200:
    for m in r.json().get("data", []):
        print("   -", m["id"])

print("\n4. Test embedding (Ollama local)...")
r = requests.post(
    LT + "/embeddings",
    headers={**hdrs, "Content-Type": "application/json"},
    json={"model": "nomic-embed-text", "input": "test embedding"},
    timeout=30,
)
print("   Status:", r.status_code)
if r.status_code == 200:
    dim = len(r.json()["data"][0]["embedding"])
    print("   Embedding dim:", dim)
else:
    print("   Error:", r.text[:200])

print("\n5. Test UI access...")
r = requests.get(LT + "/ui", allow_redirects=False)
print("   Status:", r.status_code)
print("   Location:", r.headers.get("location", "N/A"))

print("\n6. Create virtual key for UI...")
r = requests.post(
    LT + "/key/generate",
    headers={**hdrs, "Content-Type": "application/json"},
    json={"key_alias": "admin-ui-key"},
    timeout=10,
)
print("   Status:", r.status_code)
if r.status_code == 200:
    print("   Key:", r.json().get("key", "N/A")[:25], "...")
else:
    print("   Response:", r.text[:200])

print("\nDone!")
