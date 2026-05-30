# Fastembed Hybrid Qdrant Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add in-process fastembed dense+sparse embeddings and make Qdrant a first-class LiteLLM vector store provider with configurable dense / hybrid (RRF) search.

**Architecture:** A thread-safe `FastEmbedService` singleton wraps fastembed's `TextEmbedding` (dense BGE-small, 384-dim) and `SparseTextEmbedding` (SPLADE). It is exposed two ways: (1) a native `litellm.embedding` provider `fastembed/<model>` for dense, and (2) used directly by the reworked Qdrant transformation, which talks to the Qdrant 1.18 Query API for both dense and hybrid-RRF search. Models are warmed at startup and baked into the Docker image.

**Tech Stack:** Python, fastembed (ONNX), httpx, Qdrant Query API, pytest.

---

## File Structure

- Create: `litellm/llms/fastembed/__init__.py` — package marker.
- Create: `litellm/llms/fastembed/embed.py` — `FastEmbedService` singleton (dense+sparse, sync+async, warm).
- Create: `litellm/llms/fastembed/embedding_handler.py` — `litellm.embedding` dense handler returning `EmbeddingResponse`.
- Modify: `litellm/types/utils.py` — add `LlmProviders.FASTEMBED`.
- Modify: `litellm/main.py` — `embedding()` dispatch branch for `fastembed`.
- Modify: `litellm/llms/qdrant/vector_stores/transformation.py` — Query-API dense+hybrid search, named-vector create, response parse.
- Modify: `litellm/enterprise_open/vectordb/service.py` — reuse a persistent httpx client.
- Create: `litellm/vector_stores/ingest.py` — reusable Qdrant hybrid ingest helper.
- Modify: `store_countries_ollama.py` → fastembed hybrid ingest (rename target `store_countries_fastembed.py`).
- Modify: `config.local.yaml` — `countries` entry → fastembed hybrid.
- Modify: `Dockerfile` — preload fastembed models; add `fastembed` dep.
- Modify: `pyproject.toml` — add `fastembed` to proxy extras.
- Tests: `tests/test_litellm/test_fastembed_service.py`, `tests/test_litellm/test_fastembed_embedding_provider.py`, `tests/test_litellm/llm_translation/test_qdrant_vector_store.py`.

---

### Task 1: `FastEmbedService` singleton

**Files:**
- Create: `litellm/llms/fastembed/__init__.py`
- Create: `litellm/llms/fastembed/embed.py`
- Test: `tests/test_litellm/test_fastembed_service.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_litellm/test_fastembed_service.py
import sys
import types
import asyncio
import pytest


def _install_fake_fastembed(monkeypatch):
    """Install a fake `fastembed` module so tests don't download ONNX models."""
    mod = types.ModuleType("fastembed")

    class FakeDense:
        def __init__(self, model_name, **kwargs):
            self.model_name = model_name

        def embed(self, texts, **kwargs):
            for i, _ in enumerate(texts):
                yield [float(i)] * 384

    class FakeSparseEmb:
        def __init__(self, indices, values):
            self.indices = indices
            self.values = values

    class FakeSparse:
        def __init__(self, model_name, **kwargs):
            self.model_name = model_name

        def embed(self, texts, **kwargs):
            for i, _ in enumerate(texts):
                yield FakeSparseEmb([i, i + 1], [0.5, 0.25])

    mod.TextEmbedding = FakeDense
    mod.SparseTextEmbedding = FakeSparse
    monkeypatch.setitem(sys.modules, "fastembed", mod)


def test_embed_dense_shape(monkeypatch):
    _install_fake_fastembed(monkeypatch)
    from litellm.llms.fastembed.embed import FastEmbedService

    svc = FastEmbedService()
    vecs = svc.embed_dense(["hello", "world"])
    assert len(vecs) == 2
    assert len(vecs[0]) == 384


def test_embed_sparse_shape(monkeypatch):
    _install_fake_fastembed(monkeypatch)
    from litellm.llms.fastembed.embed import FastEmbedService

    svc = FastEmbedService()
    sparse = svc.embed_sparse(["hello"])
    assert sparse[0]["indices"] == [0, 1]
    assert sparse[0]["values"] == [0.5, 0.25]


def test_singleton_accessor(monkeypatch):
    _install_fake_fastembed(monkeypatch)
    from litellm.llms.fastembed.embed import get_fastembed_service

    a = get_fastembed_service()
    b = get_fastembed_service()
    assert a is b


def test_async_dense(monkeypatch):
    _install_fake_fastembed(monkeypatch)
    from litellm.llms.fastembed.embed import FastEmbedService

    svc = FastEmbedService()
    vecs = asyncio.run(svc.aembed_dense(["x"]))
    assert len(vecs) == 1 and len(vecs[0]) == 384
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_litellm/test_fastembed_service.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'litellm.llms.fastembed'`

- [ ] **Step 3: Write minimal implementation**

```python
# litellm/llms/fastembed/__init__.py
```

```python
# litellm/llms/fastembed/embed.py
"""In-process fastembed dense + sparse embedding service.

Wraps fastembed's TextEmbedding (dense) and SparseTextEmbedding (sparse/SPLADE)
in a thread-safe lazy singleton. ONNX inference is CPU-bound, so async variants
run on a thread pool executor to keep the event loop free.
"""

import asyncio
import threading
from typing import Any, Dict, List, Optional

DEFAULT_DENSE_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_SPARSE_MODEL = "prithivida/Splade_PP_en_v1"
DEFAULT_DENSE_DIM = 384


class FastEmbedService:
    """Lazy, thread-safe wrapper over fastembed dense + sparse models."""

    def __init__(
        self,
        dense_model: str = DEFAULT_DENSE_MODEL,
        sparse_model: str = DEFAULT_SPARSE_MODEL,
    ) -> None:
        self.dense_model_name = dense_model
        self.sparse_model_name = sparse_model
        self._dense: Optional[Any] = None
        self._sparse: Optional[Any] = None
        self._lock = threading.Lock()

    @staticmethod
    def _import_fastembed():
        try:
            import fastembed  # noqa: F401

            return fastembed
        except ImportError as e:
            raise ImportError(
                "fastembed is not installed. Install it with `pip install fastembed` "
                "to use in-process dense/sparse embeddings."
            ) from e

    def _ensure_dense(self) -> Any:
        if self._dense is None:
            with self._lock:
                if self._dense is None:
                    fastembed = self._import_fastembed()
                    self._dense = fastembed.TextEmbedding(self.dense_model_name)
        return self._dense

    def _ensure_sparse(self) -> Any:
        if self._sparse is None:
            with self._lock:
                if self._sparse is None:
                    fastembed = self._import_fastembed()
                    self._sparse = fastembed.SparseTextEmbedding(self.sparse_model_name)
        return self._sparse

    def warm(self) -> None:
        """Preload both models (e.g. at proxy startup)."""
        self._ensure_dense()
        self._ensure_sparse()

    def embed_dense(self, texts: List[str]) -> List[List[float]]:
        model = self._ensure_dense()
        return [list(map(float, vec)) for vec in model.embed(texts)]

    def embed_sparse(self, texts: List[str]) -> List[Dict[str, List]]:
        model = self._ensure_sparse()
        out: List[Dict[str, List]] = []
        for emb in model.embed(texts):
            out.append(
                {
                    "indices": [int(i) for i in emb.indices],
                    "values": [float(v) for v in emb.values],
                }
            )
        return out

    async def aembed_dense(self, texts: List[str]) -> List[List[float]]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_dense, texts)

    async def aembed_sparse(self, texts: List[str]) -> List[Dict[str, List]]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_sparse, texts)


_service_singleton: Optional[FastEmbedService] = None
_singleton_lock = threading.Lock()


def get_fastembed_service(
    dense_model: str = DEFAULT_DENSE_MODEL,
    sparse_model: str = DEFAULT_SPARSE_MODEL,
) -> FastEmbedService:
    """Return the process-wide FastEmbedService singleton."""
    global _service_singleton
    if _service_singleton is None:
        with _singleton_lock:
            if _service_singleton is None:
                _service_singleton = FastEmbedService(dense_model, sparse_model)
    return _service_singleton
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_litellm/test_fastembed_service.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add litellm/llms/fastembed/ tests/test_litellm/test_fastembed_service.py
git commit -m "feat: add in-process FastEmbedService (dense BGE + sparse SPLADE)"
```

---

### Task 2: Fastembed native embedding provider (dense)

**Files:**
- Modify: `litellm/types/utils.py` (add `FASTEMBED = "fastembed"` to `LlmProviders`)
- Create: `litellm/llms/fastembed/embedding_handler.py`
- Modify: `litellm/main.py` (embedding dispatch, near the `ollama` branch ~line 5447)
- Test: `tests/test_litellm/test_fastembed_embedding_provider.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_litellm/test_fastembed_embedding_provider.py
import sys
import types
import pytest


def _install_fake_fastembed(monkeypatch):
    mod = types.ModuleType("fastembed")

    class FakeDense:
        def __init__(self, model_name, **kwargs):
            self.model_name = model_name

        def embed(self, texts, **kwargs):
            for _ in texts:
                yield [0.1] * 384

    class FakeSparse:
        def __init__(self, model_name, **kwargs):
            pass

        def embed(self, texts, **kwargs):
            for _ in texts:
                yield types.SimpleNamespace(indices=[0], values=[1.0])

    mod.TextEmbedding = FakeDense
    mod.SparseTextEmbedding = FakeSparse
    monkeypatch.setitem(sys.modules, "fastembed", mod)


def test_provider_resolves():
    from litellm import get_llm_provider

    _model, provider, _key, _base = get_llm_provider(model="fastembed/BAAI/bge-small-en-v1.5")
    assert provider == "fastembed"


def test_embedding_returns_openai_shape(monkeypatch):
    _install_fake_fastembed(monkeypatch)
    # Reset singleton so the fake module is used.
    import litellm.llms.fastembed.embed as embed_mod
    embed_mod._service_singleton = None

    import litellm

    resp = litellm.embedding(model="fastembed/BAAI/bge-small-en-v1.5", input=["a", "b"])
    assert len(resp.data) == 2
    assert len(resp.data[0]["embedding"]) == 384
    assert resp.data[0]["index"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_litellm/test_fastembed_embedding_provider.py -v`
Expected: FAIL — provider not recognized / no `fastembed` branch.

- [ ] **Step 3: Write minimal implementation**

Add to `litellm/types/utils.py` in the `LlmProviders` enum (next to `QDRANT`/`S3_VECTORS`):

```python
    FASTEMBED = "fastembed"
```

Create `litellm/llms/fastembed/embedding_handler.py`:

```python
"""litellm.embedding handler for the in-process fastembed provider (dense)."""

from typing import List

from litellm.llms.fastembed.embed import get_fastembed_service
from litellm.types.utils import EmbeddingResponse, Usage


def fastembed_embedding(
    model: str,
    input: List[str],
    model_response: EmbeddingResponse,
) -> EmbeddingResponse:
    """Generate dense embeddings in-process via fastembed.

    `model` arrives without the `fastembed/` prefix (stripped by get_llm_provider),
    e.g. "BAAI/bge-small-en-v1.5".
    """
    if isinstance(input, str):
        input = [input]

    service = get_fastembed_service(dense_model=model)
    vectors = service.embed_dense(input)

    model_response.data = [
        {"object": "embedding", "index": i, "embedding": vec}
        for i, vec in enumerate(vectors)
    ]
    model_response.model = model
    model_response.object = "list"
    # fastembed does not report token usage; report input count as a proxy.
    model_response.usage = Usage(prompt_tokens=0, total_tokens=0)
    return model_response
```

Add a dispatch branch in `litellm/main.py` `embedding()` immediately after the `ollama` branch (~line 5476, before the `sagemaker` branch):

```python
        elif custom_llm_provider == "fastembed":
            from litellm.llms.fastembed.embedding_handler import fastembed_embedding

            if isinstance(input, str):
                input = [input]
            response = fastembed_embedding(
                model=model,
                input=input,
                model_response=EmbeddingResponse(),
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_litellm/test_fastembed_embedding_provider.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add litellm/types/utils.py litellm/llms/fastembed/embedding_handler.py litellm/main.py tests/test_litellm/test_fastembed_embedding_provider.py
git commit -m "feat: register fastembed as native litellm dense embedding provider"
```

---

### Task 3: Qdrant search — Query API dense + hybrid (RRF)

**Files:**
- Modify: `litellm/llms/qdrant/vector_stores/transformation.py`
- Test: `tests/test_litellm/llm_translation/test_qdrant_vector_store.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_litellm/llm_translation/test_qdrant_vector_store.py
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
    from litellm.llms.qdrant.vector_stores.transformation import QdrantVectorStoreConfig

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
    from litellm.llms.qdrant.vector_stores.transformation import QdrantVectorStoreConfig

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
    from litellm.llms.qdrant.vector_stores.transformation import QdrantVectorStoreConfig

    cfg = QdrantVectorStoreConfig()
    logging_obj = _make_logging_obj()
    logging_obj.model_call_details["litellm_params"] = {"qdrant_text_field": "text"}
    resp = MagicMock()
    resp.json.return_value = {
        "result": {
            "points": [
                {"id": "1", "score": 0.91, "payload": {"text": "Brazil", "continent": "SA"}},
                {"id": "2", "score": 0.80, "payload": {"text": "India", "continent": "Asia"}},
            ]
        }
    }
    out = cfg.transform_search_vector_store_response(resp, logging_obj)
    assert len(out["search_results"]) == 2
    assert out["search_results"][0]["content"][0]["text"] == "Brazil"
    assert out["search_results"][0]["score"] == 0.91
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_litellm/llm_translation/test_qdrant_vector_store.py -v`
Expected: FAIL — body still uses `/points/search` + `vector` key; response parser expects a list.

- [ ] **Step 3: Write minimal implementation**

Add a helper near the top of `QdrantVectorStoreConfig` (after `map_openai_params`) and replace `transform_search_vector_store_request` / `atransform_search_vector_store_request` bodies. Replace the dense-vector embedding via `litellm.embedding` with `FastEmbedService`:

```python
    def _build_query_request(
        self,
        vector_store_id: str,
        query: str,
        optional_params: dict,
        litellm_params: dict,
        dense_vec: list,
        sparse_vec: Optional[dict],
        api_base: str,
    ) -> Tuple[str, Dict[str, Any]]:
        from litellm.llms.fastembed.embed import DEFAULT_DENSE_DIM  # noqa: F401

        collection_name = vector_store_id
        url = f"{api_base}/collections/{collection_name}/points/query"
        limit = optional_params.get("limit", 5)
        mode = litellm_params.get("qdrant_search_mode", "dense")
        dense_name = litellm_params.get("qdrant_dense_vector_name", "dense")
        sparse_name = litellm_params.get("qdrant_sparse_vector_name", "sparse")

        body: Dict[str, Any] = {"limit": limit, "with_payload": True}

        if mode == "hybrid":
            body["prefetch"] = [
                {"query": dense_vec, "using": dense_name, "limit": limit},
                {"query": sparse_vec, "using": sparse_name, "limit": limit},
            ]
            body["query"] = {"fusion": "rrf"}
        else:
            body["query"] = dense_vec
            body["using"] = dense_name

        qdrant_filter = optional_params.get("filter")
        if qdrant_filter:
            body["filter"] = qdrant_filter

        score_threshold = optional_params.get(
            "score_threshold", litellm_params.get("qdrant_score_threshold")
        )
        if score_threshold is not None:
            body["score_threshold"] = score_threshold

        return url, body
```

Replace the sync `transform_search_vector_store_request` body with:

```python
        if isinstance(query, list):
            query = " ".join(query)

        from litellm.llms.fastembed.embed import (
            DEFAULT_DENSE_MODEL,
            DEFAULT_SPARSE_MODEL,
            get_fastembed_service,
        )

        mode = litellm_params.get("qdrant_search_mode", "dense")
        dense_model = litellm_params.get("fastembed_dense_model", DEFAULT_DENSE_MODEL)
        sparse_model = litellm_params.get("fastembed_sparse_model", DEFAULT_SPARSE_MODEL)
        service = get_fastembed_service(dense_model, sparse_model)

        dense_vec = service.embed_dense([query])[0]
        sparse_vec = service.embed_sparse([query])[0] if mode == "hybrid" else None

        url, request_body = self._build_query_request(
            vector_store_id, query, vector_store_search_optional_params,
            litellm_params, dense_vec, sparse_vec, api_base,
        )
        if extra_body:
            request_body.update(extra_body)
        litellm_logging_obj.model_call_details["input"] = query
        return url, request_body
```

Replace the async `atransform_search_vector_store_request` body identically but using `await service.aembed_dense([query])` and `await service.aembed_sparse([query])`.

Replace `transform_search_vector_store_response` result extraction to handle the Query API shape:

```python
            response_json = response.json()
            result = response_json.get("result", [])
            if isinstance(result, dict):
                results = result.get("points", [])
            elif isinstance(result, list):
                results = result
            else:
                results = []
```

(keep the rest of the existing parsing loop unchanged)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_litellm/llm_translation/test_qdrant_vector_store.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add litellm/llms/qdrant/vector_stores/transformation.py tests/test_litellm/llm_translation/test_qdrant_vector_store.py
git commit -m "feat: Qdrant Query-API search with configurable dense/hybrid-RRF via fastembed"
```

---

### Task 4: Qdrant create — named vectors + optimized collection

**Files:**
- Modify: `litellm/llms/qdrant/vector_stores/transformation.py` (`transform_create_vector_store_request`)
- Test: `tests/test_litellm/llm_translation/test_qdrant_vector_store.py` (add cases)

- [ ] **Step 1: Write the failing test**

```python
def test_create_named_vectors_hybrid():
    from litellm.llms.qdrant.vector_stores.transformation import QdrantVectorStoreConfig

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
    from litellm.llms.qdrant.vector_stores.transformation import QdrantVectorStoreConfig

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_litellm/llm_translation/test_qdrant_vector_store.py -k create -v`
Expected: FAIL — current create builds an unnamed `vectors` dict.

- [ ] **Step 3: Write minimal implementation**

Replace `transform_create_vector_store_request` body:

```python
        params = vector_store_create_optional_params
        collection_name = params.get("vector_store_name", "default")
        url = f"{api_base}/collections/{collection_name}"

        vector_size = params.get("qdrant_vector_size", 384)
        distance = params.get("qdrant_distance", "Cosine")
        dense_name = params.get("qdrant_dense_vector_name", "dense")
        sparse_name = params.get("qdrant_sparse_vector_name", "sparse")
        mode = params.get("qdrant_search_mode", "dense")

        request_body: Dict[str, Any] = {
            "vectors": {
                dense_name: {
                    "size": vector_size,
                    "distance": distance,
                    "hnsw_config": {"m": 16, "ef_construct": 100},
                }
            }
        }
        if mode == "hybrid":
            request_body["sparse_vectors"] = {sparse_name: {}}

        collection_config = params.get("qdrant_collection_config", {})
        if collection_config:
            request_body.update(collection_config)

        return url, request_body
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_litellm/llm_translation/test_qdrant_vector_store.py -k create -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add litellm/llms/qdrant/vector_stores/transformation.py tests/test_litellm/llm_translation/test_qdrant_vector_store.py
git commit -m "feat: Qdrant named-vector collection create with HNSW + optional sparse"
```

---

### Task 5: Ingest helper + demo script + config

**Files:**
- Create: `litellm/vector_stores/ingest.py`
- Create: `store_countries_fastembed.py` (adapted from `store_countries_ollama.py`)
- Modify: `config.local.yaml`

- [ ] **Step 1: Write the ingest helper**

```python
# litellm/vector_stores/ingest.py
"""Helper to (re)create a Qdrant hybrid collection and upsert points with
dense (BGE) + sparse (SPLADE) vectors generated in-process via fastembed."""

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
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["api-key"] = api_key
    base = api_base.rstrip("/")
    with httpx.Client(timeout=60) as client:
        client.delete(f"{base}/collections/{collection}", headers=headers)
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
        resp = client.put(f"{base}/collections/{collection}", headers=headers, json=body)
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
    """`docs` is a list of payload dicts that each contain `text_field`."""
    service = get_fastembed_service()
    texts = [d[text_field] for d in docs]
    dense = service.embed_dense(texts)
    sparse = service.embed_sparse(texts)

    points = []
    for i, doc in enumerate(docs):
        points.append(
            {
                "id": i + 1,
                "vector": {dense_name: dense[i], sparse_name: sparse[i]},
                "payload": doc,
            }
        )

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
```

- [ ] **Step 2: Adapt the demo script**

Copy `store_countries_ollama.py` to `store_countries_fastembed.py`; replace its embedding + upsert logic with calls to `recreate_hybrid_collection` and `upsert_documents` from `litellm.vector_stores.ingest`. Keep the same country dataset and the demo queries, but route search through the proxy `/v1/vector_stores/countries/search`.

- [ ] **Step 3: Update config.local.yaml**

Replace the `countries` registry entry with:

```yaml
vector_store_registry:
  - vector_store_name: "countries"
    litellm_params:
      vector_store_id: "countries"
      custom_llm_provider: "qdrant"
      api_base: "http://localhost:6333"
      qdrant_search_mode: "hybrid"
      fastembed_dense_model: "BAAI/bge-small-en-v1.5"
      fastembed_sparse_model: "prithivida/Splade_PP_en_v1"
      qdrant_dense_vector_name: "dense"
      qdrant_sparse_vector_name: "sparse"
      qdrant_vector_size: 384
      qdrant_distance: "Cosine"
      qdrant_text_field: "text"
```

Remove the `nomic-embed-text` (ollama) model entry from `model_list` if it is no longer referenced.

- [ ] **Step 4: Commit**

```bash
git add litellm/vector_stores/ingest.py store_countries_fastembed.py config.local.yaml
git commit -m "feat: fastembed hybrid ingest helper + countries demo + config"
```

---

### Task 6: Startup warm, httpx reuse, deps, Dockerfile

**Files:**
- Modify: `litellm/enterprise_open/vectordb/service.py` (persistent client + warm hook reuse)
- Modify: `Dockerfile`
- Modify: `pyproject.toml`

- [ ] **Step 1: Reuse a persistent httpx client in the vectordb service**

In `VectorDBService.__init__`, add `self._client: Optional[httpx.AsyncClient] = None` and a helper:

```python
    def _get_client(self, timeout: int) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=timeout)
        return self._client
```

Replace the `async with httpx.AsyncClient(...) as client:` blocks with `client = self._get_client(cfg.timeout)` (do not close per-call). Add an `async def aclose(self)` that closes `self._client` for shutdown.

- [ ] **Step 2: Add the fastembed preload to the Dockerfile**

After the application dependencies are installed in `Dockerfile`, add:

```dockerfile
# Pre-download fastembed ONNX models so they are baked into the image
RUN python -c "from fastembed import TextEmbedding, SparseTextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5'); SparseTextEmbedding('prithivida/Splade_PP_en_v1')"
```

- [ ] **Step 3: Add fastembed to dependencies**

In `pyproject.toml`, add `fastembed` to the proxy optional-dependency group (the `proxy` extra). Then locally:

```bash
uv pip install fastembed
```

- [ ] **Step 4: Verify the service still imports**

Run: `uv run python -c "from litellm.enterprise_open.vectordb.service import VectorDBService; print('ok')"`
Expected: `ok`

- [ ] **Step 5: Commit**

```bash
git add litellm/enterprise_open/vectordb/service.py Dockerfile pyproject.toml
git commit -m "perf: warm fastembed at startup, reuse httpx client, bake models into image"
```

---

### Task 7: Full verification

- [ ] **Step 1: Format**

Run: `uv run black litellm/llms/fastembed litellm/llms/qdrant litellm/vector_stores/ingest.py tests/test_litellm/test_fastembed_service.py tests/test_litellm/test_fastembed_embedding_provider.py tests/test_litellm/llm_translation/test_qdrant_vector_store.py`

- [ ] **Step 2: Run the new test suite**

Run: `uv run pytest tests/test_litellm/test_fastembed_service.py tests/test_litellm/test_fastembed_embedding_provider.py tests/test_litellm/llm_translation/test_qdrant_vector_store.py -v`
Expected: all PASS

- [ ] **Step 3: Enterprise import smoke**

Run: `uv run python -c "import litellm.enterprise_open.router; import litellm.enterprise_open.vectordb.service; print('enterprise ok')"`
Expected: `enterprise ok`

- [ ] **Step 4: (Manual, with Qdrant running) end-to-end demo**

Run: `uv run python store_countries_fastembed.py`
Expected: collection recreated, hybrid search returns Brazil ranked #1 for "largest country in South America".

- [ ] **Step 5: Commit any formatting**

```bash
git add -A
git commit -m "chore: format + verify fastembed hybrid qdrant"
```
