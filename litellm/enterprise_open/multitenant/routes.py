"""
Multi-tenant management API routes.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request

from litellm.enterprise_open.config import TenantConfig
from litellm.enterprise_open.multitenant.middleware import (
    TenantContext,
    TenantScopedDataAccess,
    extract_tenant_context,
)

logger = logging.getLogger(__name__)
tenant_router = APIRouter()


@tenant_router.get("/me")
async def get_my_tenant(request: Request):
    """Get the current user's tenant context."""
    ctx = await extract_tenant_context(request)
    return {"tenant": ctx.to_dict()}


@tenant_router.get("/usage")
async def get_tenant_usage(
    request: Request,
    period: Optional[str] = Query("30d", description="Usage period: 7d, 30d, 90d"),
):
    """Get usage metrics for the current tenant."""
    ctx = await extract_tenant_context(request)

    try:
        from litellm.proxy.proxy_server import prisma_client
    except ImportError:
        raise HTTPException(status_code=503, detail="Database not configured")

    if prisma_client is None:
        raise HTTPException(status_code=503, detail="Database not configured")

    access = TenantScopedDataAccess(ctx)

    # Count resources
    counts = await access.count_resources(prisma_client)

    # Calculate total spend
    where = access._apply_tenant_filter({})
    spend_logs = await prisma_client.db.litellm_spendlogs.find_many(where=where)
    total_spend = sum(float(getattr(log, "spend", 0) or 0) for log in spend_logs)
    total_requests = len(spend_logs)

    return {
        "tenant": ctx.to_dict(),
        "resources": counts,
        "usage": {
            "total_requests": total_requests,
            "total_spend": round(total_spend, 6),
            "period": period,
        },
    }


@tenant_router.get("/users")
async def list_tenant_users(
    request: Request,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
):
    """List users in the current tenant."""
    ctx = await extract_tenant_context(request)

    try:
        from litellm.proxy.proxy_server import prisma_client
    except ImportError:
        raise HTTPException(status_code=503, detail="Database not configured")

    if prisma_client is None:
        raise HTTPException(status_code=503, detail="Database not configured")

    access = TenantScopedDataAccess(ctx)
    users = await access.list_users(prisma_client, page=page, page_size=page_size)
    return {
        "users": [
            {
                "user_id": getattr(u, "user_id", None),
                "user_email": getattr(u, "user_email", None),
                "user_role": getattr(u, "user_role", None),
            }
            for u in users
        ],
        "page": page,
        "page_size": page_size,
    }


@tenant_router.get("/teams")
async def list_tenant_teams(
    request: Request,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
):
    """List teams in the current tenant."""
    ctx = await extract_tenant_context(request)

    try:
        from litellm.proxy.proxy_server import prisma_client
    except ImportError:
        raise HTTPException(status_code=503, detail="Database not configured")

    if prisma_client is None:
        raise HTTPException(status_code=503, detail="Database not configured")

    access = TenantScopedDataAccess(ctx)
    teams = await access.list_teams(prisma_client, page=page, page_size=page_size)
    return {
        "teams": [
            {
                "team_id": getattr(t, "team_id", None),
                "team_alias": getattr(t, "team_alias", None),
                "organization_id": getattr(t, "organization_id", None),
            }
            for t in teams
        ],
        "page": page,
        "page_size": page_size,
    }


@tenant_router.get("/spend")
async def get_tenant_spend(
    request: Request,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
):
    """Get spend logs for the current tenant."""
    ctx = await extract_tenant_context(request)

    try:
        from litellm.proxy.proxy_server import prisma_client
    except ImportError:
        raise HTTPException(status_code=503, detail="Database not configured")

    if prisma_client is None:
        raise HTTPException(status_code=503, detail="Database not configured")

    access = TenantScopedDataAccess(ctx)
    logs = await access.list_spend_logs(prisma_client, page=page, page_size=page_size)
    return {
        "spend_logs": [
            {
                "model": getattr(l, "model", None),
                "spend": getattr(l, "spend", None),
                "prompt_tokens": getattr(l, "prompt_tokens", None),
                "completion_tokens": getattr(l, "completion_tokens", None),
                "start_time": str(getattr(l, "startTime", None) or ""),
            }
            for l in logs
        ],
        "page": page,
        "page_size": page_size,
    }
