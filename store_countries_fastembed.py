#!/usr/bin/env python3
"""
Store 10 countries in Qdrant with in-process fastembed hybrid vectors.

Dense (BAAI/bge-small-en-v1.5, 384-dim) + sparse (prithivida/Splade_PP_en_v1)
are generated locally via FastEmbedService — no external embedding service.
Search is demonstrated both directly against the Qdrant Query API (RRF hybrid)
and through the LiteLLM proxy vector store search endpoint.
"""

import os
import sys

import httpx

from litellm.llms.fastembed.embed import get_fastembed_service
from litellm.vector_stores.ingest import recreate_hybrid_collection, upsert_documents

QDRANT = os.environ.get("QDRANT_API_BASE", "http://localhost:6333")
LITELLM = os.environ.get("LITELLM_API_BASE", "http://localhost:4000")
MASTER_KEY = os.environ.get("LITELLM_MASTER_KEY", "")
COLLECTION = "countries"

countries = [
    {
        "country": "India",
        "capital": "New Delhi",
        "continent": "Asia",
        "population": "1.4 billion",
        "text": "India is a country in South Asia with its capital New Delhi. It has a population of approximately 1.4 billion people.",
    },
    {
        "country": "United States",
        "capital": "Washington D.C.",
        "continent": "North America",
        "population": "331 million",
        "text": "The United States of America is located in North America with its capital Washington D.C. It has a population of approximately 331 million people.",
    },
    {
        "country": "Japan",
        "capital": "Tokyo",
        "continent": "Asia",
        "population": "125 million",
        "text": "Japan is an island country in East Asia with its capital Tokyo. It has a population of approximately 125 million people.",
    },
    {
        "country": "Brazil",
        "capital": "Brasilia",
        "continent": "South America",
        "population": "214 million",
        "text": "Brazil is the largest country in South America with its capital Brasilia. It has a population of approximately 214 million people.",
    },
    {
        "country": "Germany",
        "capital": "Berlin",
        "continent": "Europe",
        "population": "83 million",
        "text": "Germany is a country in Central Europe with its capital Berlin. It has a population of approximately 83 million people.",
    },
    {
        "country": "Australia",
        "capital": "Canberra",
        "continent": "Oceania",
        "population": "26 million",
        "text": "Australia is a country and continent in Oceania with its capital Canberra. It has a population of approximately 26 million people.",
    },
    {
        "country": "France",
        "capital": "Paris",
        "continent": "Europe",
        "population": "67 million",
        "text": "France is a country in Western Europe with its capital Paris. It has a population of approximately 67 million people.",
    },
    {
        "country": "South Korea",
        "capital": "Seoul",
        "continent": "Asia",
        "population": "52 million",
        "text": "South Korea is a country in East Asia with its capital Seoul. It has a population of approximately 52 million people.",
    },
    {
        "country": "Egypt",
        "capital": "Cairo",
        "continent": "Africa",
        "population": "104 million",
        "text": "Egypt is a transcontinental country in Africa and Asia with its capital Cairo. It has a population of approximately 104 million people.",
    },
    {
        "country": "Canada",
        "capital": "Ottawa",
        "continent": "North America",
        "population": "38 million",
        "text": "Canada is a country in North America with its capital Ottawa. It has a population of approximately 38 million people.",
    },
]


def step(msg):
    print(f"\n{'=' * 60}\n  {msg}\n{'=' * 60}")


def log(msg):
    print(f"  {msg}")


def hybrid_search(query, limit=3, continent=None):
    """Direct Qdrant Query API hybrid (dense + sparse, RRF fusion)."""
    service = get_fastembed_service()
    dense_vec = service.embed_dense([query])[0]
    sparse_vec = service.embed_sparse([query])[0]
    body = {
        "prefetch": [
            {"query": dense_vec, "using": "dense", "limit": limit},
            {"query": sparse_vec, "using": "sparse", "limit": limit},
        ],
        "query": {"fusion": "rrf"},
        "limit": limit,
        "with_payload": True,
    }
    if continent:
        body["filter"] = {"must": [{"key": "continent", "match": {"value": continent}}]}
    resp = httpx.post(
        f"{QDRANT}/collections/{COLLECTION}/points/query", json=body, timeout=30
    )
    resp.raise_for_status()
    return resp.json()["result"]["points"]


def main():
    step("STEP 1: Warm fastembed models (dense BGE + sparse SPLADE)")
    get_fastembed_service().warm()
    log("✅ models loaded in-process")

    step("STEP 2: (Re)create hybrid 'countries' collection (384-dim dense + sparse)")
    recreate_hybrid_collection(QDRANT, COLLECTION, vector_size=384)
    log("✅ collection created with named dense + sparse vectors")

    step("STEP 3: Embed + upsert 10 countries (dense + sparse)")
    upsert_documents(QDRANT, COLLECTION, countries)
    log(f"✅ upserted {len(countries)} points")

    info = httpx.get(f"{QDRANT}/collections/{COLLECTION}", timeout=10).json()["result"]
    log(f"✅ points_count={info['points_count']} status={info['status']}")

    step("STEP 4: Hybrid search demos (dense + sparse, RRF)")
    queries = [
        "Which country has the capital Paris?",
        "What is the largest country in South America?",
        "Tell me about Asian countries",
    ]
    for query in queries:
        log(f'\n  Query: "{query}"')
        for j, hit in enumerate(hybrid_search(query)):
            p = hit["payload"]
            log(
                f"  {j + 1}. {p['country']} -> {p['capital']} (score: {hit['score']:.4f}) [{p['continent']}]"
            )

    step("STEP 5: Filtered hybrid search — 'large population' in Asia only")
    for j, hit in enumerate(
        hybrid_search("large population country", limit=5, continent="Asia")
    ):
        p = hit["payload"]
        log(
            f"  {j + 1}. {p['country']} -> {p['capital']} (score: {hit['score']:.4f}) [pop: {p['population']}]"
        )

    step("STEP 6: Search via LiteLLM proxy vector store endpoint")
    headers = {"Content-Type": "application/json"}
    if MASTER_KEY:
        headers["Authorization"] = f"Bearer {MASTER_KEY}"
    try:
        resp = httpx.post(
            f"{LITELLM}/v1/vector_stores/{COLLECTION}/search",
            headers=headers,
            json={"query": "largest country in South America", "max_num_results": 3},
            timeout=30,
        )
        if resp.status_code == 200:
            for j, r in enumerate(resp.json().get("data", [])):
                text = r["content"][0]["text"] if r.get("content") else ""
                log(f"  {j + 1}. score={r.get('score')} {text[:60]}")
        else:
            log(f"⚠️  proxy returned HTTP {resp.status_code}: {resp.text[:160]}")
    except Exception as e:
        log(f"⚠️  proxy not reachable ({e}) — start it with `python start_proxy.py`")

    step("SUMMARY")
    log("✅ 10 countries stored in Qdrant with fastembed hybrid vectors")
    log(
        "✅ dense: BAAI/bge-small-en-v1.5 (384-dim) | sparse: prithivida/Splade_PP_en_v1"
    )
    log("✅ in-process embeddings — zero external embedding API calls")
    log("✅ Qdrant dashboard: http://localhost:6333/dashboard")


if __name__ == "__main__":
    sys.exit(main())
