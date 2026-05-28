"""
Audit log API routes — query, filter, aggregate, and export.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from litellm.enterprise_open.audit.service import get_audit_service

logger = logging.getLogger(__name__)
audit_router = APIRouter()


@audit_router.get("/logs")
async def query_audit_logs(
    request: Request,
    action: Optional[str] = Query(None),
    user_id: Optional[str] = Query(None),
    table_name: Optional[str] = Query(None),
    object_id: Optional[str] = Query(None),
    start_date: Optional[str] = Query(None, description="ISO datetime"),
    end_date: Optional[str] = Query(None, description="ISO datetime"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    sort_by: str = Query("created_at"),
    sort_order: str = Query("desc"),
):
    """Query audit logs with filters and pagination."""
    service = get_audit_service()
    try:
        result = await service.query_logs(
            action=action,
            user_id=user_id,
            table_name=table_name,
            object_id=object_id,
            start_date=start_date,
            end_date=end_date,
            page=page,
            page_size=page_size,
            sort_by=sort_by,
            sort_order=sort_order,
        )
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


@audit_router.get("/logs/{log_id}")
async def get_audit_log(log_id: str, request: Request):
    """Get a single audit log entry by ID."""
    service = get_audit_service()
    log = await service.get_log(log_id)
    if not log:
        raise HTTPException(status_code=404, detail=f"Audit log '{log_id}' not found")
    return log


@audit_router.get("/summary")
async def audit_summary(
    request: Request,
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    group_by: str = Query("action", description="Group by: action, table_name, changed_by"),
):
    """Get aggregated audit log statistics."""
    service = get_audit_service()
    try:
        return await service.get_summary(
            start_date=start_date,
            end_date=end_date,
            group_by=group_by,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


@audit_router.get("/export")
async def export_audit_logs(
    request: Request,
    format: str = Query("json", description="Export format: json or csv"),
    action: Optional[str] = Query(None),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
):
    """Export audit logs to JSON or CSV."""
    if format not in ("json", "csv"):
        raise HTTPException(status_code=400, detail="Format must be 'json' or 'csv'")

    service = get_audit_service()
    try:
        content = await service.export_logs(
            format=format,
            action=action,
            start_date=start_date,
            end_date=end_date,
        )
        if format == "csv":
            return PlainTextResponse(
                content=content,
                media_type="text/csv",
                headers={"Content-Disposition": "attachment; filename=audit_logs.csv"},
            )
        return PlainTextResponse(
            content=content,
            media_type="application/json",
            headers={"Content-Disposition": "attachment; filename=audit_logs.json"},
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
