"""
Vector DB Management API Routes.

REST endpoints for managing Qdrant, Milvus, and pgvector backends
through the LiteLLM proxy.

All endpoints under /enterprise/vectordb/
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request

from litellm.enterprise_open.vectordb.config import (
    EnterpriseVectorDBConfig,
    VectorDBBackend,
    VectorDBCollectionInfo,
    VectorDBHealthStatus,
)
from litellm.enterprise_open.vectordb.service import VectorDBService

vectordb_router = APIRouter(prefix="/vectordb", tags=["Vector DB Management"])


def _get_service(request: Request) -> VectorDBService:
    """Get VectorDBService from app state."""
    service = getattr(request.app.state, "enterprise_vectordb_service", None)
    if not service:
        raise HTTPException(
            status_code=503,
            detail="Vector DB service not initialized. Configure vectordb in enterprise_settings.",
        )
    return service


@vectordb_router.get("/health", response_model=Dict[str, VectorDBHealthStatus])
async def get_all_health(request: Request):
    """Check health of all configured vector DB backends."""
    service = _get_service(request)
    statuses = await service.check_all_health()
    return {s.backend.value: s.model_dump() for s in statuses}


@vectordb_router.get("/health/{backend}", response_model=VectorDBHealthStatus)
async def get_backend_health(request: Request, backend: str):
    """Check health of a specific vector DB backend."""
    service = _get_service(request)
    try:
        vb = VectorDBBackend(backend)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown backend: {backend}. Supported: qdrant, milvus, pgvector",
        )
    status = await service.check_health(vb)
    return status.model_dump()


@vectordb_router.get("/collections", response_model=List[VectorDBCollectionInfo])
async def list_all_collections(request: Request, backend: Optional[str] = None):
    """List collections across all or a specific backend."""
    service = _get_service(request)
    if backend:
        try:
            vb = VectorDBBackend(backend)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Unknown backend: {backend}")
        return await service.list_collections(vb)

    results = []
    for name in service.config.backends:
        try:
            vb = VectorDBBackend(name)
            collections = await service.list_collections(vb)
            results.extend(collections)
        except ValueError:
            continue
    return results


@vectordb_router.get("/config", response_model=Dict[str, Any])
async def get_vectordb_config(request: Request):
    """Get current vector DB configuration (secrets masked)."""
    service = _get_service(request)
    config = service.config

    # Mask sensitive fields
    result = {
        "backends": {},
        "routing": config.routing.model_dump(),
    }
    for name, cfg in config.backends.items():
        masked = dict(cfg)
        if "api_key" in masked and masked["api_key"]:
            masked["api_key"] = "***masked***"
        if "database_url" in masked and masked["database_url"]:
            masked["database_url"] = "***masked***"
        result["backends"][name] = masked

    return result


@vectordb_router.post("/routing/resolve")
async def resolve_backend(
    request: Request,
    model: Optional[str] = None,
    team_id: Optional[str] = None,
):
    """Resolve which vector DB backend should handle a request."""
    service = _get_service(request)
    backend = service.resolve_backend(model=model, team_id=team_id)
    return {
        "backend": backend.value,
        "model": model,
        "team_id": team_id,
    }
