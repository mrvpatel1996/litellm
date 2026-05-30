# Fastembed In-Process Hybrid Embeddings + Qdrant Native Vector Store Provider

**Date:** 2026-05-30
**Branch:** enterprise-features
**Status:** Approved

## Goal

Replace the Ollama embedding network hop with **in-process fastembed** producing both
**dense** (`BAAI/bge-small-en-v1.5`, 384-dim) and **sparse** (`prithivida/Splade_PP_en_v1`,
SPLADE) vectors, and make **Qdrant** a first-class LiteLLM vector store provider with a
**configurable dense / hybrid (RRF)** search mode. Optimize for latency and cold-start.

Milvus and S3 Vectors already exist upstream and are out of scope except where shared
plumbing touches them. Enterprise features (`enterprise_open/`) must keep working but are
not the focus.

## Decisions (locked)

- **Hybrid mode:** configurable per collection — `qdrant_search_mode` is `"dense"` (default)
  or `"hybrid"`. Hybrid fuses dense + sparse server-side via Reciprocal Rank Fusion (RRF)
  through the Qdrant 1.18 Query API.
- **Fastembed wiring:** a shared in-process `FastEmbedService` singleton (dense + sparse,
  warmed at startup, async via executor) AND a native `litellm.embedding` provider
  `fastembed/<model>` for dense embeddings everywhere.
- **Scope:** Qdrant is the provider brought to full quality. Milvus/S3 untouched.
- **Re-ingest:** the `countries` demo collection is recreated at 384-dim + sparse; the prior
  768-dim Ollama data is replaced.

## Components

### 1. `FastEmbedService` — `litellm/llms/fastembed/embed.py`
- Thread-safe lazy singleton holding `TextEmbedding(dense_model)` and
  `SparseTextEmbedding(sparse_model)`.
- `embed_dense(texts) -> List[List[float]]`
- `embed_sparse(texts) -> List[{"indices": List[int], "values": List[float]}]`
- `aembed_dense` / `aembed_sparse` via `loop.run_in_executor` (ONNX is CPU-bound; keep the
  event loop free).
- `warm()` preloads both models; called at proxy startup.
- Defaults: dense `BAAI/bge-small-en-v1.5` (384), sparse `prithivida/Splade_PP_en_v1`.
  Configurable via constructor / env.
- `fastembed` is an optional dependency; import is lazy with a clear error if missing.

### 2. Fastembed native embedding provider (dense)
- Add `LlmProviders.FASTEMBED = "fastembed"` (`litellm/types/utils.py`).
- `get_llm_provider` resolves `model="fastembed/BAAI/bge-small-en-v1.5"`.
- `embedding()` / `aembedding()` dispatch a `fastembed` branch that calls
  `FastEmbedService.embed_dense` and returns a standard `EmbeddingResponse` (with usage).
- Sparse output is NOT exposed through `litellm.embedding` (that interface is dense-only);
  sparse/hybrid goes through `FastEmbedService` directly.

### 3. Qdrant provider — `litellm/llms/qdrant/vector_stores/transformation.py`
- Use the Qdrant **Query API** (`POST /collections/{c}/points/query`) uniformly.
- New `litellm_params`: `qdrant_search_mode` (`dense`|`hybrid`), `fastembed_dense_model`,
  `fastembed_sparse_model`, `qdrant_dense_vector_name` (`dense`),
  `qdrant_sparse_vector_name` (`sparse`), `qdrant_vector_size` (384),
  `qdrant_score_threshold`, `qdrant_text_field`.
- **dense mode:** embed query via `FastEmbedService.embed_dense` →
  `{"query": vec, "using": <dense_name>}`.
- **hybrid mode:** `prefetch` over named dense + named sparse →
  `{"query": {"fusion": "rrf"}}`. Single round-trip.
- **create:** named-vector collection — `vectors.{dense}` (size, distance) plus
  `sparse_vectors.{sparse}` when hybrid. Optimized HNSW (`m`, `ef_construct`), optional
  scalar quantization for dense, on-disk payload, payload indexes for filtered fields.
- **response parse:** handle Query API shape `result.points[]` (id, score, payload).
- Keep sync + async paths in parity.

### 4. Ingestion + config
- A small reusable ingest helper plus updated `store_countries_*.py`: create the hybrid
  collection and upsert points with both named vectors via `FastEmbedService`.
- `config.local.yaml`: `countries` registry entry switched to fastembed hybrid; remove the
  Ollama embedding model from the embedding path.

### 5. Optimizations
- Warm `FastEmbedService` at proxy startup (mirrors the Dockerfile model preload).
- Reuse a persistent `httpx.AsyncClient` in `enterprise_open/vectordb/service.py` instead of
  per-call clients.
- Add `fastembed` to proxy dependencies.
- Add to `Dockerfile`:
  `RUN python -c "from fastembed import TextEmbedding, SparseTextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5'); SparseTextEmbedding('prithivida/Splade_PP_en_v1')"`
  so models are baked into the image (no cold-start download).

### 6. Tests — `tests/test_litellm/`
- `FastEmbedService` with fastembed mocked: dense/sparse shapes, singleton identity, async.
- Qdrant transform: dense request body, hybrid prefetch+RRF body, create named-vectors body,
  Query API response parse, text-field projection, filter passthrough.
- Provider resolution for `fastembed` and `qdrant`.

### 7. Enterprise features
- No behavior change. `enterprise_open/vectordb` keeps loading and reuses `FastEmbedService`.
  Confirm imports still resolve.

## Data flow (search)

```
proxy /v1/vector_stores/{id}/search
  → QdrantVectorStoreConfig.atransform_search_vector_store_request
     → FastEmbedService.aembed_dense(query)            (+ aembed_sparse if hybrid)
     → POST /collections/{id}/points/query
          dense:  {"query": vec, "using": "dense"}
          hybrid: {"prefetch": [dense, sparse], "query": {"fusion": "rrf"}}
  → transform_search_vector_store_response (result.points[])
```

## Non-goals
- No hybrid for Milvus/S3 in this pass.
- No new enterprise capabilities.
- No change to the OpenAI-compatible vector store API surface.

## Risks
- `fastembed` not installed → clear ImportError with install hint; provider/service guarded.
- Named-vector schema differs from the current unnamed `countries` collection → handled by
  re-ingest.
- Query API requires Qdrant ≥ 1.10 (server is 1.18.1 — OK).
