"""
Vector DB Service Layer.

Unified interface for managing Qdrant, Milvus, and pgvector backends.
Handles health checks, collection management, and routing.
"""

import time
from typing import Any, Dict, List, Optional

import httpx

from litellm.enterprise_open.vectordb.config import (
    EnterpriseVectorDBConfig,
    VectorDBBackend,
    VectorDBCollectionInfo,
    VectorDBHealthStatus,
)


class VectorDBService:
    """Unified service for managing multiple vector DB backends."""

    def __init__(self, config: EnterpriseVectorDBConfig):
        self.config = config
        self._health_cache: Dict[str, VectorDBHealthStatus] = {}
        self._last_health_check: Dict[str, float] = {}

    # ─── Health Checks ─────────────────────────────────────────────

    async def check_health(
        self, backend: VectorDBBackend
    ) -> VectorDBHealthStatus:
        """Check health of a specific backend."""
        start = time.time()
        try:
            if backend == VectorDBBackend.QDRANT:
                return await self._check_qdrant_health(start)
            elif backend == VectorDBBackend.MILVUS:
                return await self._check_milvus_health(start)
            elif backend == VectorDBBackend.PGVECTOR:
                return await self._check_pgvector_health(start)
            else:
                return VectorDBHealthStatus(
                    backend=backend,
                    connected=False,
                    error=f"Unknown backend: {backend}",
                )
        except Exception as e:
            return VectorDBHealthStatus(
                backend=backend,
                connected=False,
                error=str(e),
                latency_ms=(time.time() - start) * 1000,
            )

    async def check_all_health(self) -> List[VectorDBHealthStatus]:
        """Check health of all configured backends."""
        results = []
        for backend_name in self.config.backends:
            try:
                backend = VectorDBBackend(backend_name)
            except ValueError:
                continue
            status = await self.check_health(backend)
            results.append(status)
            self._health_cache[backend_name] = status
            self._last_health_check[backend_name] = time.time()
        return results

    async def _check_qdrant_health(self, start: float) -> VectorDBHealthStatus:
        cfg = self.config.get_qdrant_config()
        if not cfg:
            return VectorDBHealthStatus(
                backend=VectorDBBackend.QDRANT,
                connected=False,
                error="Qdrant not configured",
            )
        async with httpx.AsyncClient(timeout=cfg.timeout) as client:
            headers = {}
            if cfg.api_key:
                headers["api-key"] = cfg.api_key
            resp = await client.get(f"{cfg.api_base}/healthz", headers=headers)
            latency = (time.time() - start) * 1000
            if resp.status_code == 200:
                title = resp.json().get("title", "")
                version = resp.json().get("version", "")
                # Get collections count
                collections_resp = await client.get(
                    f"{cfg.api_base}/collections", headers=headers
                )
                collections_count = None
                if collections_resp.status_code == 200:
                    collections_count = len(
                        collections_resp.json().get("result", {}).get("collections", [])
                    )
                return VectorDBHealthStatus(
                    backend=VectorDBBackend.QDRANT,
                    connected=True,
                    latency_ms=round(latency, 2),
                    collections_count=collections_count,
                    version=version or title,
                )
            return VectorDBHealthStatus(
                backend=VectorDBBackend.QDRANT,
                connected=False,
                error=f"HTTP {resp.status_code}",
                latency_ms=round(latency, 2),
            )

    async def _check_milvus_health(self, start: float) -> VectorDBHealthStatus:
        cfg = self.config.get_milvus_config()
        if not cfg:
            return VectorDBHealthStatus(
                backend=VectorDBBackend.MILVUS,
                connected=False,
                error="Milvus not configured",
            )
        async with httpx.AsyncClient(timeout=cfg.timeout) as client:
            headers = {"Content-Type": "application/json"}
            if cfg.api_key:
                headers["Authorization"] = f"Bearer {cfg.api_key}"
            resp = await client.get(
                f"{cfg.api_base}/v2/vectordb/collections/list",
                headers=headers,
            )
            latency = (time.time() - start) * 1000
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                return VectorDBHealthStatus(
                    backend=VectorDBBackend.MILVUS,
                    connected=True,
                    latency_ms=round(latency, 2),
                    collections_count=len(data) if isinstance(data, list) else None,
                )
            return VectorDBHealthStatus(
                backend=VectorDBBackend.MILVUS,
                connected=False,
                error=f"HTTP {resp.status_code}",
                latency_ms=round(latency, 2),
            )

    async def _check_pgvector_health(self, start: float) -> VectorDBHealthStatus:
        cfg = self.config.get_pgvector_config()
        if not cfg:
            return VectorDBHealthStatus(
                backend=VectorDBBackend.PGVECTOR,
                connected=False,
                error="pgvector not configured",
            )
        async with httpx.AsyncClient(timeout=30) as client:
            headers = {"Content-Type": "application/json"}
            if cfg.api_key:
                headers["Authorization"] = f"Bearer {cfg.api_key}"
            resp = await client.get(
                f"{cfg.api_base}/v1/vector_stores",
                headers=headers,
            )
            latency = (time.time() - start) * 1000
            if resp.status_code in (200, 201):
                return VectorDBHealthStatus(
                    backend=VectorDBBackend.PGVECTOR,
                    connected=True,
                    latency_ms=round(latency, 2),
                )
            return VectorDBHealthStatus(
                backend=VectorDBBackend.PGVECTOR,
                connected=False,
                error=f"HTTP {resp.status_code}",
                latency_ms=round(latency, 2),
            )

    # ─── Collection Management ─────────────────────────────────────

    async def list_collections(
        self, backend: VectorDBBackend
    ) -> List[VectorDBCollectionInfo]:
        """List all collections/indexes for a backend."""
        if backend == VectorDBBackend.QDRANT:
            return await self._list_qdrant_collections()
        elif backend == VectorDBBackend.MILVUS:
            return await self._list_milvus_collections()
        elif backend == VectorDBBackend.PGVECTOR:
            return await self._list_pgvector_collections()
        return []

    async def _list_qdrant_collections(self) -> List[VectorDBCollectionInfo]:
        cfg = self.config.get_qdrant_config()
        if not cfg:
            return []
        async with httpx.AsyncClient(timeout=cfg.timeout) as client:
            headers = {}
            if cfg.api_key:
                headers["api-key"] = cfg.api_key
            resp = await client.get(
                f"{cfg.api_base}/collections", headers=headers
            )
            if resp.status_code != 200:
                return []
            collections = resp.json().get("result", {}).get("collections", [])
            results = []
            for col in collections:
                name = col.get("name", "")
                # Get collection info
                info_resp = await client.get(
                    f"{cfg.api_base}/collections/{name}", headers=headers
                )
                if info_resp.status_code == 200:
                    info = info_resp.json().get("result", {})
                    config = info.get("config", {})
                    params = config.get("params", {})
                    vectors = params.get("vectors", {})
                    vector_size = vectors.get("size", 0)
                    distance = vectors.get("distance", "Unknown")
                    points_count = info.get("points_count", 0)
                    status = info.get("status", "unknown")
                    results.append(
                        VectorDBCollectionInfo(
                            name=name,
                            backend=VectorDBBackend.QDRANT,
                            vector_size=vector_size,
                            distance=distance,
                            points_count=points_count,
                            status=status,
                        )
                    )
            return results

    async def _list_milvus_collections(self) -> List[VectorDBCollectionInfo]:
        cfg = self.config.get_milvus_config()
        if not cfg:
            return []
        async with httpx.AsyncClient(timeout=cfg.timeout) as client:
            headers = {"Content-Type": "application/json"}
            if cfg.api_key:
                headers["Authorization"] = f"Bearer {cfg.api_key}"
            resp = await client.post(
                f"{cfg.api_base}/v2/vectordb/collections/list",
                headers=headers,
                json={},
            )
            if resp.status_code != 200:
                return []
            data = resp.json().get("data", [])
            results = []
            for col in data:
                name = col.get("collectionName", col.get("name", ""))
                results.append(
                    VectorDBCollectionInfo(
                        name=name,
                        backend=VectorDBBackend.MILVUS,
                        vector_size=0,  # Need describe call
                        distance="Unknown",
                        metadata=col,
                    )
                )
            return results

    async def _list_pgvector_collections(self) -> List[VectorDBCollectionInfo]:
        cfg = self.config.get_pgvector_config()
        if not cfg:
            return []
        async with httpx.AsyncClient(timeout=30) as client:
            headers = {"Content-Type": "application/json"}
            if cfg.api_key:
                headers["Authorization"] = f"Bearer {cfg.api_key}"
            resp = await client.get(
                f"{cfg.api_base}/v1/vector_stores",
                headers=headers,
            )
            if resp.status_code != 200:
                return []
            data = resp.json().get("data", [])
            results = []
            for store in data:
                results.append(
                    VectorDBCollectionInfo(
                        name=store.get("id", ""),
                        backend=VectorDBBackend.PGVECTOR,
                        vector_size=cfg.embedding_dim,
                        distance="Cosine",
                        metadata=store,
                    )
                )
            return results

    # ─── Routing ───────────────────────────────────────────────────

    def resolve_backend(
        self,
        model: Optional[str] = None,
        team_id: Optional[str] = None,
    ) -> VectorDBBackend:
        """Resolve which backend to use based on routing config."""
        routing = self.config.routing

        # Check team routing first
        if team_id and team_id in routing.team_routing:
            return routing.team_routing[team_id]

        # Check model routing
        if model:
            for pattern, backend in routing.model_routing.items():
                if pattern.endswith("*"):
                    if model.startswith(pattern[:-1]):
                        return backend
                elif model == pattern:
                    return backend

        # Default
        return routing.default_backend
