import sys
import types
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def fake_fastembed(monkeypatch):
    mod = types.ModuleType("fastembed")

    class FakeDense:
        def __init__(self, model_name, **kwargs):
            pass

        def embed(self, texts, **kwargs):
            for _ in texts:
                yield [0.1] * 384

    class FakeSparse:
        def __init__(self, model_name, **kwargs):
            pass

        def embed(self, texts, **kwargs):
            for _ in texts:
                yield types.SimpleNamespace(indices=[1, 7], values=[0.9, 0.4])

    mod.TextEmbedding = FakeDense
    mod.SparseTextEmbedding = FakeSparse
    monkeypatch.setitem(sys.modules, "fastembed", mod)
    import litellm.llms.fastembed.embed as embed_mod

    embed_mod._service_singleton = None


def _make_logging_obj():
    obj = MagicMock()
    obj.model_call_details = {"litellm_params": {}}
    return obj


def test_dense_query_body(fake_fastembed):
    from litellm.llms.qdrant.vector_stores.transformation import (
        QdrantVectorStoreConfig,
    )

    cfg = QdrantVectorStoreConfig()
    url, body = cfg.transform_search_vector_store_request(
        vector_store_id="countries",
        query="largest country",
        vector_store_search_optional_params={"limit": 3},
        api_base="http://localhost:6333",
        litellm_logging_obj=_make_logging_obj(),
        litellm_params={
            "qdrant_search_mode": "dense",
            "fastembed_dense_model": "BAAI/bge-small-en-v1.5",
            "qdrant_dense_vector_name": "dense",
        },
    )
    assert url.endswith("/collections/countries/points/query")
    assert body["using"] == "dense"
    assert len(body["query"]) == 384
    assert body["limit"] == 3
    assert body["with_payload"] is True


def test_hybrid_query_body(fake_fastembed):
    from litellm.llms.qdrant.vector_stores.transformation import (
        QdrantVectorStoreConfig,
    )

    cfg = QdrantVectorStoreConfig()
    url, body = cfg.transform_search_vector_store_request(
        vector_store_id="countries",
        query="largest country",
        vector_store_search_optional_params={"limit": 5},
        api_base="http://localhost:6333",
        litellm_logging_obj=_make_logging_obj(),
        litellm_params={
            "qdrant_search_mode": "hybrid",
            "fastembed_dense_model": "BAAI/bge-small-en-v1.5",
            "fastembed_sparse_model": "prithivida/Splade_PP_en_v1",
            "qdrant_dense_vector_name": "dense",
            "qdrant_sparse_vector_name": "sparse",
        },
    )
    assert url.endswith("/collections/countries/points/query")
    assert body["query"] == {"fusion": "rrf"}
    prefetch = body["prefetch"]
    assert len(prefetch) == 2
    dense_pf = next(p for p in prefetch if p["using"] == "dense")
    sparse_pf = next(p for p in prefetch if p["using"] == "sparse")
    assert len(dense_pf["query"]) == 384
    assert sparse_pf["query"] == {"indices": [1, 7], "values": [0.9, 0.4]}


def test_query_api_response_parse():
    from litellm.llms.qdrant.vector_stores.transformation import (
        QdrantVectorStoreConfig,
    )

    cfg = QdrantVectorStoreConfig()
    logging_obj = _make_logging_obj()
    logging_obj.model_call_details["litellm_params"] = {"qdrant_text_field": "text"}
    resp = MagicMock()
    resp.json.return_value = {
        "result": {
            "points": [
                {
                    "id": "1",
                    "score": 0.91,
                    "payload": {"text": "Brazil", "continent": "SA"},
                },
                {
                    "id": "2",
                    "score": 0.80,
                    "payload": {"text": "India", "continent": "Asia"},
                },
            ]
        }
    }
    out = cfg.transform_search_vector_store_response(resp, logging_obj)
    assert len(out["search_results"]) == 2
    assert out["search_results"][0]["content"][0]["text"] == "Brazil"
    assert out["search_results"][0]["score"] == 0.91


def test_create_named_vectors_hybrid():
    from litellm.llms.qdrant.vector_stores.transformation import (
        QdrantVectorStoreConfig,
    )

    cfg = QdrantVectorStoreConfig()
    url, body = cfg.transform_create_vector_store_request(
        vector_store_create_optional_params={
            "vector_store_name": "countries",
            "qdrant_vector_size": 384,
            "qdrant_distance": "Cosine",
            "qdrant_search_mode": "hybrid",
            "qdrant_dense_vector_name": "dense",
            "qdrant_sparse_vector_name": "sparse",
        },
        api_base="http://localhost:6333",
    )
    assert url.endswith("/collections/countries")
    assert body["vectors"]["dense"]["size"] == 384
    assert body["vectors"]["dense"]["distance"] == "Cosine"
    assert "sparse" in body["sparse_vectors"]


def test_create_named_vectors_dense_only():
    from litellm.llms.qdrant.vector_stores.transformation import (
        QdrantVectorStoreConfig,
    )

    cfg = QdrantVectorStoreConfig()
    url, body = cfg.transform_create_vector_store_request(
        vector_store_create_optional_params={
            "vector_store_name": "docs",
            "qdrant_vector_size": 384,
        },
        api_base="http://localhost:6333",
    )
    assert body["vectors"]["dense"]["size"] == 384
    assert "sparse_vectors" not in body
