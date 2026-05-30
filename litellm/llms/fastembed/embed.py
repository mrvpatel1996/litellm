"""In-process fastembed dense + sparse embedding service.

Wraps fastembed's ``TextEmbedding`` (dense) and ``SparseTextEmbedding``
(sparse / SPLADE) in a thread-safe lazy singleton. ONNX inference is
CPU-bound, so the async variants run on a thread-pool executor to keep the
event loop free.
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
    def _import_fastembed() -> Any:
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
        return [[float(x) for x in vec] for vec in model.embed(texts)]

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
    """Return the process-wide :class:`FastEmbedService` singleton."""
    global _service_singleton
    if _service_singleton is None:
        with _singleton_lock:
            if _service_singleton is None:
                _service_singleton = FastEmbedService(dense_model, sparse_model)
    return _service_singleton
