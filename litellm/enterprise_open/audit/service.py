"""
Enhanced audit log service — query, filter, aggregate, and export.
"""

import csv
import io
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class AuditLogService:
    """Service for querying and managing audit logs via Prisma."""

    def __init__(self):
        self._prisma = None

    def _get_prisma(self):
        """Lazy-load Prisma client."""
        if self._prisma is None:
            from litellm.proxy.proxy_server import prisma_client
            self._prisma = prisma_client
        return self._prisma

    async def query_logs(
        self,
        action: Optional[str] = None,
        user_id: Optional[str] = None,
        table_name: Optional[str] = None,
        object_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        page: int = 1,
        page_size: int = 50,
        sort_by: str = "created_at",
        sort_order: str = "desc",
    ) -> Dict[str, Any]:
        """
        Query audit logs with filters and pagination.

        Returns:
            Dict with 'logs' list, 'total' count, 'page', 'page_size'
        """
        prisma = self._get_prisma()
        if prisma is None:
            raise RuntimeError("Database not configured")

        where: Dict[str, Any] = {}
        if action:
            where["action"] = action
        if user_id:
            where["changed_by"] = user_id
        if table_name:
            where["table_name"] = table_name
        if object_id:
            where["object_id"] = object_id
        if start_date or end_date:
            created_at_filter: Dict[str, Any] = {}
            if start_date:
                created_at_filter["gte"] = start_date
            if end_date:
                created_at_filter["lte"] = end_date
            where["created_at"] = created_at_filter

        # Count total matching logs
        total = await prisma.db.litellm_auditlog.count(where=where)

        # Calculate pagination
        skip = (page - 1) * page_size

        # Determine sort
        order = "desc" if sort_order == "desc" else "asc"
        order_by = {sort_by: order}

        # Fetch logs
        logs = await prisma.db.litellm_auditlog.find_many(
            where=where,
            order=order_by,
            skip=skip,
            take=page_size,
        )

        # Serialize
        log_list = []
        for log in logs:
            log_dict = {}
            for field in ["id", "action", "table_name", "object_id", "changed_by",
                          "changed_by_api_key", "before_value", "after_value",
                          "created_at", "updated_at"]:
                val = getattr(log, field, None)
                if val is not None:
                    if isinstance(val, datetime):
                        val = val.isoformat()
                    log_dict[field] = val
            log_list.append(log_dict)

        return {
            "logs": log_list,
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size if total > 0 else 0,
        }

    async def get_log(self, log_id: str) -> Optional[Dict[str, Any]]:
        """Get a single audit log entry by ID."""
        prisma = self._get_prisma()
        if prisma is None:
            raise RuntimeError("Database not configured")

        log = await prisma.db.litellm_auditlog.find_unique(where={"id": log_id})
        if not log:
            return None

        log_dict = {}
        for field in ["id", "action", "table_name", "object_id", "changed_by",
                      "changed_by_api_key", "before_value", "after_value",
                      "created_at", "updated_at"]:
            val = getattr(log, field, None)
            if val is not None:
                if isinstance(val, datetime):
                    val = val.isoformat()
                log_dict[field] = val
        return log_dict

    async def get_summary(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        group_by: str = "action",
    ) -> Dict[str, Any]:
        """
        Get aggregated audit log statistics.

        Args:
            start_date: Filter start date
            end_date: Filter end date
            group_by: Group by 'action', 'table_name', or 'changed_by'
        """
        prisma = self._get_prisma()
        if prisma is None:
            raise RuntimeError("Database not configured")

        where: Dict[str, Any] = {}
        if start_date or end_date:
            created_at_filter: Dict[str, Any] = {}
            if start_date:
                created_at_filter["gte"] = start_date
            if end_date:
                created_at_filter["lte"] = end_date
            where["created_at"] = created_at_filter

        # Get all matching logs (for aggregation — Prisma doesn't have groupBy in this version)
        logs = await prisma.db.litellm_auditlog.find_many(where=where)

        # Aggregate
        counts: Dict[str, int] = {}
        for log in logs:
            key = getattr(log, group_by, "unknown") or "unknown"
            counts[key] = counts.get(key, 0) + 1

        # Sort by count descending
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)

        return {
            "total_events": len(logs),
            "group_by": group_by,
            "start_date": start_date,
            "end_date": end_date,
            "breakdown": [{"key": k, "count": v} for k, v in sorted_counts],
        }

    async def export_logs(
        self,
        format: str = "json",
        action: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> str:
        """
        Export audit logs to JSON or CSV.

        Returns:
            String content in the requested format
        """
        result = await self.query_logs(
            action=action,
            start_date=start_date,
            end_date=end_date,
            page_size=10000,  # Export up to 10K logs
        )
        logs = result["logs"]

        if format == "csv":
            output = io.StringIO()
            if logs:
                writer = csv.DictWriter(output, fieldnames=logs[0].keys())
                writer.writeheader()
                writer.writerows(logs)
            return output.getvalue()
        else:
            return json.dumps(logs, indent=2, default=str)

    async def log_event(
        self,
        action: str,
        table_name: str,
        object_id: str,
        changed_by: str,
        before_value: Optional[str] = None,
        after_value: Optional[str] = None,
        changed_by_api_key: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Create a new audit log entry.

        Returns:
            Created log entry dict
        """
        prisma = self._get_prisma()
        if prisma is None:
            logger.warning("Database not configured, audit log not saved")
            return {"status": "skipped", "reason": "no database"}

        log = await prisma.db.litellm_auditlog.create(
            data={
                "action": action,
                "table_name": table_name,
                "object_id": object_id,
                "changed_by": changed_by,
                "before_value": before_value,
                "after_value": after_value,
                "changed_by_api_key": changed_by_api_key,
            }
        )
        return {"status": "ok", "id": log.id}


# Singleton
_service: Optional[AuditLogService] = None


def get_audit_service() -> AuditLogService:
    global _service
    if _service is None:
        _service = AuditLogService()
    return _service
