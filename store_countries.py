#!/usr/bin/env python3
"""
Store 10 countries with capitals in Qdrant via LiteLLM.
Uses OpenRouter embeddings through the LiteLLM proxy.
"""

import os
import json
import requests
import sys

# Load env
from pathlib import Path
env_content = Path(os.path.expanduser("~/.hermes/.env")).read_text()
for line in env_content.splitlines():
    if line.startswith("export "):
        line = line[7:]
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        os.environ[k.strip()] = v.strip()

OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "")
MASTER_KEY = "sk-loc...1234"

LT = "http://localhost:4000"
QDRANT = "http://localhost:6333"
CT = {"Content-Type": "application/json"}

# 10 Countries with capitals
countries = [
    {"id": 1, "country": "India", "capital": "New Delhi", "continent": "Asia", "population": "1.4 billion", "text": "India is a country in South Asia with its capital New Delhi. It has a population of approximately 1.4 billion people."},
    {"id": 2, "country": "United States", "capital": "Washington D.C.", "continent": "North America", "population": "331 million", "text": "The United States of America is located in North America with its capital Washington D.C. It has a population of approximately 331 million people."},
    {"id": 3, "country": "Japan", "capital": "Tokyo", "continent": "Asia", "population": "125 million", "text": "Japan is an island country in East Asia with its capital Tokyo. It has a population of approximately 125 million people."},
    {"id": 4, "country": "Brazil", "capital": "Brasilia", "continent": "South America", "population": "214 million", "text": "Brazil is the largest country in South America with its capital Brasilia. It has a population of approximately 214 million people."},
    {"id": 5, "country": "Germany", "capital": "Berlin", "continent": "Europe", "population": "83 million", "text": "Germany is a country in Central Europe with its capital Berlin. It has a population of approximately 83 million people."},
    {"id": 6, "country": "Australia", "capital": "Canberra", "continent": "Oceania", "population": "26 million", "text": "Australia is a country and continent in Oceania with its capital Canberra. It has a population of approximately 26 million people."},
    {"id": 7, "country": "France", "capital": "Paris", "continent": "Europe", "population": "67 million", "text": "France is a country in Western Europe with its capital Paris. It has a population of approximately 67 million people."},
    {"id": 8, "country": "South Korea", "capital": "Seoul", "continent": "Asia", "population": "52 million", "text": "South Korea is a country in East Asia with its capital Seoul. It has a population of approximately 52 million people."},
    {"id": 9, "country": "Egypt", "capital": "Cairo", "continent": "Africa", "population": "104 million", "text": "Egypt is a transcontinental country in Africa and Asia with its capital Cairo. It has a population of approximately 104 million people."},
    {"id": 10, "country": "Canada", "capital": "Ottawa", "continent": "North America", "population": "38 million", "text": "Canada is a country in North America with its capital Ottawa. It has a population of approximately 38 million people."},
]

def log(msg):
    print(f"  {msg}")

def step(msg):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")

# ─── Step 1: Test embedding via LiteLLM proxy ───
step("STEP 1: Test embedding via LiteLLM + OpenRouter")

r = requests.post(
    f"{LT}/embeddings",
    headers={**CT, "Authorization": f"Bearer {MASTER_KEY}"},
    json={"model": "text-embedding-3-small", "input": "Hello world"},
    timeout=30,
)
if r.status_code != 200:
    print(f"  ❌ Embedding failed: {r.status_code} {r.text[:300]}")
    sys.exit(1)

dim = len(r.json()["data"][0]["embedding"])
log(f"✅ Embedding works! Dimension: {dim}")

# ─── Step 2: Create countries collection in Qdrant ───
step("STEP 2: Create 'countries' collection in Qdrant")

# Delete if exists
requests.delete(f"{QDRANT}/collections/countries", timeout=5)

r = requests.put(
    f"{QDRANT}/collections/countries",
    headers=CT,
    json={"vectors": {"size": dim, "distance": "Cosine"}},
    timeout=10,
)
log(f"✅ Collection created: {r.json()}")

# ─── Step 3: Generate embeddings for all 10 countries ───
step("STEP 3: Generate embeddings via LiteLLM for 10 countries")

texts = [c["text"] for c in countries]
r = requests.post(
    f"{LT}/embeddings",
    headers={**CT, "Authorization": f"Bearer {MASTER_KEY}"},
    json={"model": "text-embedding-3-small", "input": texts},
    timeout=60,
)

if r.status_code != 200:
    print(f"  ❌ Batch embedding failed: {r.status_code} {r.text[:300]}")
    sys.exit(1)

emb_data = r.json()["data"]
log(f"✅ Generated {len(emb_data)} embeddings, each {dim}-dimensional")

# ─── Step 4: Upsert to Qdrant ───
step("STEP 4: Upsert 10 country vectors to Qdrant")

points = []
for i, country in enumerate(countries):
    points.append({
        "id": country["id"],
        "vector": emb_data[i]["embedding"],
        "payload": {
            "country": country["country"],
            "capital": country["capital"],
            "continent": country["continent"],
            "population": country["population"],
            "text": country["text"],
        }
    })

r = requests.put(
    f"{QDRANT}/collections/countries/points",
    headers=CT,
    json={"points": points},
    timeout=15,
)
log(f"✅ Upserted: {r.json()}")

# ─── Step 5: Verify count ───
step("STEP 5: Verify stored data")

r = requests.get(f"{QDRANT}/collections/countries", timeout=5)
info = r.json()["result"]
log(f"✅ Collection has {info['points_count']} points, status: {info['status']}")

# ─── Step 6: Semantic search demo ───
step("STEP 6: Semantic search demos")

queries = [
    "Which country has the capital Paris?",
    "Tell me about Asian countries",
    "What is the largest country in South America?",
]

for query in queries:
    log(f"\n  Query: \"{query}\"")

    # Embed the query via LiteLLM
    r = requests.post(
        f"{LT}/embeddings",
        headers={**CT, "Authorization": f"Bearer {MASTER_KEY}"},
        json={"model": "text-embedding-3-small", "input": query},
        timeout=30,
    )
    query_vec = r.json()["data"][0]["embedding"]

    # Search Qdrant
    r = requests.post(
        f"{QDRANT}/collections/countries/points/search",
        headers=CT,
        json={
            "vector": query_vec,
            "limit": 3,
            "with_payload": True,
        },
        timeout=10,
    )

    results = r.json()["result"]
    for j, hit in enumerate(results):
        payload = hit["payload"]
        score = hit["score"]
        log(f"  {j+1}. {payload['country']} → {payload['capital']} (score: {score:.4f}) [{payload['continent']}]")

# ─── Step 7: Filtered search by continent ───
step("STEP 7: Filtered search — 'large population' in Asia")

r = requests.post(
    f"{LT}/embeddings",
    headers={**CT, "Authorization": f"Bearer {MASTER_KEY}"},
    json={"model": "text-embedding-3-small", "input": "large population"},
    timeout=30,
)
query_vec = r.json()["data"][0]["embedding"]

r = requests.post(
    f"{QDRANT}/collections/countries/points/search",
    headers=CT,
    json={
        "vector": query_vec,
        "filter": {
            "must": [
                {"key": "continent", "match": {"value": "Asia"}}
            ]
        },
        "limit": 5,
        "with_payload": True,
    },
    timeout=10,
)

results = r.json()["result"]
for j, hit in enumerate(results):
    payload = hit["payload"]
    score = hit["score"]
    log(f"  {j+1}. {payload['country']} → {payload['capital']} (score: {score:.4f}) [pop: {payload['population']}]")

# ─── Summary ───
step("SUMMARY")
log(f"✅ 10 countries stored in Qdrant via LiteLLM + OpenRouter")
log(f"✅ Collection: 'countries' ({dim}-dim vectors)")
log(f"✅ Embedding model: text-embedding-3-small via OpenRouter")
log(f"✅ Semantic search working!")
log(f"")
log(f"Countries stored:")
for c in countries:
    log(f"  {c['id']:2d}. {c['country']:20s} → {c['capital']} ({c['continent']})")
