"""Helpers to (re)create a Qdrant hybrid collection and upsert points.

Dense (BGE-small) and sparse (SPLADE) vectors are generated in-process via
:class:`litellm.llms.fastembed.embed.FastEmbedService`, so no external
embedding service is required.
"""

from typing import Any, Dict, List, Optional

import httpx

from litellm.llms.fastembed.embed import DEFAULT_DENSE_DIM, get_fastembed_service


def recreate_hybrid_collection(
    api_base: str,
    collection: str,
    dense_name: str = "dense",
    sparse_name: str = "sparse",
    vector_size: int = DEFAULT_DENSE_DIM,
    distance: str = "Cosine",
    api_key: Optional[str] = None,
) -> None:
    """Drop and recreate a Qdrant collection with named dense + sparse vectors."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["api-key"] = api_key
    base = api_base.rstrip("/")
    body = {
        "vectors": {
            dense_name: {
                "size": vector_size,
                "distance": distance,
                "hnsw_config": {"m": 16, "ef_construct": 100},
            }
        },
        "sparse_vectors": {sparse_name: {}},
    }
    with httpx.Client(timeout=60) as client:
        client.delete(f"{base}/collections/{collection}", headers=headers)
        resp = client.put(
            f"{base}/collections/{collection}", headers=headers, json=body
        )
        resp.raise_for_status()


def upsert_documents(
    api_base: str,
    collection: str,
    docs: List[Dict[str, Any]],
    text_field: str = "text",
    dense_name: str = "dense",
    sparse_name: str = "sparse",
    api_key: Optional[str] = None,
) -> None:
    """Embed and upsert ``docs`` (payload dicts each containing ``text_field``)."""
    service = get_fastembed_service()
    texts = [doc[text_field] for doc in docs]
    dense = service.embed_dense(texts)
    sparse = service.embed_sparse(texts)

    points = [
        {
            "id": i + 1,
            "vector": {dense_name: dense[i], sparse_name: sparse[i]},
            "payload": doc,
        }
        for i, doc in enumerate(docs)
    ]

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["api-key"] = api_key
    base = api_base.rstrip("/")
    with httpx.Client(timeout=120) as client:
        resp = client.put(
            f"{base}/collections/{collection}/points?wait=true",
            headers=headers,
            json={"points": points},
        )
        resp.raise_for_status()
