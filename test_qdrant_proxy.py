#!/usr/bin/env python3
"""Test Qdrant vector store search via LiteLLM proxy."""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

BASE = "http://localhost:4000"
KEY = os.getenv("LITELLM_MASTER_KEY")
HEADERS = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}

print("=" * 60)
print("  Testing Qdrant via LiteLLM Proxy")
print("=" * 60)

# 1. List vector stores
print("\n1. GET /vector_store/list")
r = requests.get(f"{BASE}/vector_store/list", headers=HEADERS)
print(f"   Status: {r.status_code}")
data = r.json()
print(f"   Count: {data.get('total_count', 0)}")
for vs in data.get("data", []):
    print(f"   - {vs.get('vector_store_id')}: {vs.get('custom_llm_provider')}")

# 2. Check available routes
print("\n2. Checking available vector store routes...")
r = requests.get(f"{BASE}/openapi.json", headers=HEADERS)
if r.status_code == 200:
    paths = r.json().get("paths", {})
    vs_routes = [p for p in paths if "vector" in p.lower()]
    print(f"   Vector store routes found: {vs_routes}")

# 3. Try search via v1 endpoint
print("\n3. POST /v1/vector_stores/search (OpenAI-compatible)")
r = requests.post(f"{BASE}/v1/vector_stores/search", headers=HEADERS, json={
    "vector_store_id": "countries",
    "query": "capital of France",
    "limit": 3
})
print(f"   Status: {r.status_code}")
print(f"   Response: {r.text[:500]}")

# 4. Try direct Qdrant search through LiteLLM embedding + Qdrant REST
print("\n4. Direct test: embed via proxy -> search Qdrant")
# First embed
r = requests.post(f"{BASE}/embeddings", headers=HEADERS, json={
    "model": "nomic-embed-text",
    "input": "capital of France"
})
print(f"   Embed status: {r.status_code}")
if r.status_code == 200:
    vector = r.json()["data"][0]["embedding"]
    print(f"   Vector dim: {len(vector)}")
    
    # Search Qdrant directly
    r2 = requests.post("http://localhost:6333/collections/countries/points/search", 
        json={"vector": vector, "limit": 3, "with_payload": True})
    print(f"   Qdrant search status: {r2.status_code}")
    results = r2.json().get("result", [])
    for hit in results:
        payload = hit.get("payload", {})
        print(f"   - {payload.get('country', '?')} -> {payload.get('capital', '?')} (score: {hit.get('score', 0):.4f})")

print("\n" + "=" * 60)
print("  Done!")
print("=" * 60)
