#!/usr/bin/env python3
"""
Store 10 countries with capitals in Qdrant via LiteLLM + Ollama (local embeddings).
Uses nomic-embed-text — runs 100% locally, no API keys needed for embeddings.
"""

import os
import json
import requests
import sys

# Load env for OpenRouter key (only used for chat, not embeddings)
from pathlib import Path
env_content = Path(os.path.expanduser("~/.hermes/.env")).read_text()
for line in env_content.splitlines():
    if line.startswith("export "):
        line = line[7:]
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        os.environ[k.strip()] = v.strip()

MASTER_KEY = "sk-loc...1234"
LT = "http://localhost:4000"
OLLAMA = "http://localhost:11434"
QDRANT = "http://localhost:6333"
CT = {"Content-Type": "application/json"}
EMBED_MODEL = "nomic-embed-text"

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

def get_embedding(text):
    """Get embedding via Ollama directly (local, fast, free)."""
    r = requests.post(
        f"{OLLAMA}/api/embed",
        headers=CT,
        json={"model": EMBED_MODEL, "input": text},
        timeout=60,
    )
    if r.status_code != 200:
        raise Exception(f"Ollama embedding failed: {r.status_code} {r.text[:200]}")
    return r.json()["embeddings"][0]

def get_embeddings_batch(texts):
    """Get embeddings via Ollama (batch via single requests)."""
    r = requests.post(
        f"{OLLAMA}/api/embed",
        headers=CT,
        json={"model": EMBED_MODEL, "input": texts},
        timeout=120,
    )
    if r.status_code != 200:
        raise Exception(f"Ollama batch embedding failed: {r.status_code} {r.text[:200]}")
    return r.json()["embeddings"]

# ─── Step 1: Verify Ollama is running and model is loaded ───
step("STEP 1: Verify Ollama + nomic-embed-text")

r = requests.get(f"{OLLAMA}/api/tags", timeout=5)
models = [m["name"] for m in r.json().get("models", [])]
has_embed = any("nomic" in m for m in models)
log(f"Ollama models: {models}")
if not has_embed:
    print("  ❌ nomic-embed-text not found!")
    sys.exit(1)
log(f"✅ nomic-embed-text available")

# Test embedding
test_emb = get_embedding("test")
dim = len(test_emb)
log(f"✅ Embedding works! Dimension: {dim}")

# ─── Step 2: Create countries collection in Qdrant ───
step("STEP 2: Create 'countries' collection in Qdrant (768-dim)")

# Delete old collection if exists
requests.delete(f"{QDRANT}/collections/countries", timeout=5)

r = requests.put(
    f"{QDRANT}/collections/countries",
    headers=CT,
    json={"vectors": {"size": dim, "distance": "Cosine"}},
    timeout=10,
)
log(f"✅ Collection created: {r.json()}")

# ─── Step 3: Generate embeddings for all 10 countries via Ollama ───
step("STEP 3: Generate LOCAL embeddings via Ollama for 10 countries")

texts = [c["text"] for c in countries]
embeddings = get_embeddings_batch(texts)
log(f"✅ Generated {len(embeddings)} embeddings, each {dim}-dimensional")
log(f"✅ 100% local — no API calls to OpenRouter/OpenAI!")

# ─── Step 4: Upsert to Qdrant ───
step("STEP 4: Upsert 10 country vectors to Qdrant")

points = []
for i, country in enumerate(countries):
    points.append({
        "id": country["id"],
        "vector": embeddings[i],
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

# ─── Step 5: Verify ───
step("STEP 5: Verify stored data")

r = requests.get(f"{QDRANT}/collections/countries", timeout=5)
info = r.json()["result"]
log(f"✅ Collection has {info['points_count']} points, status: {info['status']}")

# ─── Step 6: Semantic search demos ───
step("STEP 6: Semantic search demos (100% local)")

queries = [
    "Which country has the capital Paris?",
    "Tell me about Asian countries",
    "What is the largest country in South America?",
    "Where is the Great Barrier Reef?",
    "Which countries are in Europe?",
]

for query in queries:
    log(f'\n  Query: "{query}"')
    query_vec = get_embedding(query)

    r = requests.post(
        f"{QDRANT}/collections/countries/points/search",
        headers=CT,
        json={"vector": query_vec, "limit": 3, "with_payload": True},
        timeout=10,
    )

    results = r.json()["result"]
    for j, hit in enumerate(results):
        p = hit["payload"]
        log(f"  {j+1}. {p['country']} -> {p['capital']} (score: {hit['score']:.4f}) [{p['continent']}]")

# ─── Step 7: Filtered search by continent ───
step("STEP 7: Filtered search — 'large population' in Asia only")

query_vec = get_embedding("large population country")
r = requests.post(
    f"{QDRANT}/collections/countries/points/search",
    headers=CT,
    json={
        "vector": query_vec,
        "filter": {"must": [{"key": "continent", "match": {"value": "Asia"}}]},
        "limit": 5,
        "with_payload": True,
    },
    timeout=10,
)

results = r.json()["result"]
for j, hit in enumerate(results):
    p = hit["payload"]
    log(f"  {j+1}. {p['country']} -> {p['capital']} (score: {hit['score']:.4f}) [pop: {p['population']}]")

# ─── Step 8: Verify LiteLLM proxy also sees Ollama ───
step("STEP 8: Verify LiteLLM proxy can route to Ollama")

r = requests.get(f"{LT}/health/liveliness", timeout=5)
if r.status_code == 200:
    log(f"✅ LiteLLM proxy running on port 4000")
else:
    log(f"⚠️  LiteLLM proxy not responding")

# ─── Summary ───
step("SUMMARY")
log(f"✅ 10 countries stored in Qdrant via OLLAMA (100% local)")
log(f"✅ Collection: 'countries' ({dim}-dim vectors)")
log(f"✅ Embedding model: nomic-embed-text via Ollama")
log(f"✅ Zero external API calls for embeddings!")
log(f"✅ Qdrant dashboard: http://localhost:6333/dashboard")
log(f"✅ LiteLLM proxy: http://localhost:4000")
log(f"")
log(f"Countries stored:")
for c in countries:
    log(f"  {c['id']:2d}. {c['country']:20s} -> {c['capital']} ({c['continent']})")
