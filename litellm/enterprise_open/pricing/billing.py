"""
Invoice generation service — creates invoices from spend data + pricing rules.
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_INVOICES_PATH = os.path.expanduser("~/.litellm/invoices")


class InvoiceService:
    """Generates and manages invoices from spend data."""

    def __init__(self, invoices_path: str = DEFAULT_INVOICES_PATH):
        self._path = Path(invoices_path)
        self._path.mkdir(parents=True, exist_ok=True)

    def _invoice_path(self, invoice_id: str) -> Path:
        return self._path / f"{invoice_id}.json"

    async def generate_invoice(
        self,
        organization_id: Optional[str] = None,
        team_id: Optional[str] = None,
        period_start: Optional[str] = None,
        period_end: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate an invoice from spend logs + pricing rules.

        Returns:
            Invoice dict with line items and totals
        """
        from litellm.enterprise_open.pricing.engine import get_pricing_engine

        try:
            from litellm.proxy.proxy_server import prisma_client
        except ImportError:
            raise RuntimeError("Database not configured")

        if prisma_client is None:
            raise RuntimeError("Database not configured")

        # Build filter
        where: Dict[str, Any] = {}
        if organization_id:
            # Filter by org members
            where["org_id"] = organization_id
        if team_id:
            where["team_id"] = team_id
        if period_start or period_end:
            date_filter: Dict[str, Any] = {}
            if period_start:
                date_filter["gte"] = period_start
            if period_end:
                date_filter["lte"] = period_end
            where["startTime"] = date_filter

        # Fetch spend logs
        spend_logs = await prisma_client.db.litellm_spendlogs.find_many(
            where=where,
            order={"startTime": "asc"},
        )

        if not spend_logs:
            return {
                "status": "no_data",
                "message": "No spend data found for the specified period",
            }

        # Group spend by model and apply pricing rules
        pricing_engine = get_pricing_engine()
        model_spend: Dict[str, Dict] = {}
        total_base = 0.0
        total_markup = 0.0
        total_final = 0.0

        for log in spend_logs:
            model = getattr(log, "model", "unknown") or "unknown"
            spend = float(getattr(log, "spend", 0) or 0)

            if model not in model_spend:
                model_spend[model] = {
                    "model": model,
                    "requests": 0,
                    "base_cost": 0.0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                }

            model_spend[model]["requests"] += 1
            model_spend[model]["base_cost"] += spend
            model_spend[model]["input_tokens"] += int(
                getattr(log, "prompt_tokens", 0) or 0
            )
            model_spend[model]["output_tokens"] += int(
                getattr(log, "completion_tokens", 0) or 0
            )

        # Apply pricing rules to each model group
        line_items = []
        for model, data in model_spend.items():
            pricing = pricing_engine.calculate_cost(
                model=model,
                base_cost=data["base_cost"],
                input_tokens=data["input_tokens"],
                output_tokens=data["output_tokens"],
                organization_id=organization_id,
                team_id=team_id,
            )
            line_item = {
                **data,
                "markup": pricing["markup"],
                "flat_fee": pricing.get("flat_fee", 0),
                "custom_cost": pricing["custom_cost"],
                "final_cost": pricing["total"],
                "applied_rule": pricing["applied_rule"],
            }
            line_items.append(line_item)
            total_base += data["base_cost"]
            total_markup += pricing["markup"]
            total_final += pricing["total"]

        # Create invoice record
        import uuid
        invoice_id = f"inv-{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc).isoformat()

        invoice = {
            "invoice_id": invoice_id,
            "organization_id": organization_id,
            "team_id": team_id,
            "period_start": period_start,
            "period_end": period_end,
            "created_at": now,
            "status": "generated",
            "line_items": line_items,
            "summary": {
                "total_requests": sum(item["requests"] for item in line_items),
                "total_base_cost": round(total_base, 6),
                "total_markup": round(total_markup, 6),
                "total_final_cost": round(total_final, 6),
                "models_used": len(line_items),
            },
        }

        # Save to disk
        with open(self._invoice_path(invoice_id), "w") as f:
            json.dump(invoice, f, indent=2, default=str)

        logger.info(
            f"Generated invoice {invoice_id}: ${total_final:.4f} "
            f"({len(line_items)} models, {invoice['summary']['total_requests']} requests)"
        )
        return invoice

    def list_invoices(self) -> List[Dict]:
        """List all saved invoices."""
        invoices = []
        for path in sorted(self._path.glob("inv-*.json"), reverse=True):
            try:
                with open(path) as f:
                    data = json.load(f)
                invoices.append({
                    "invoice_id": data["invoice_id"],
                    "organization_id": data.get("organization_id"),
                    "team_id": data.get("team_id"),
                    "period_start": data.get("period_start"),
                    "period_end": data.get("period_end"),
                    "created_at": data.get("created_at"),
                    "status": data.get("status"),
                    "total_final_cost": data.get("summary", {}).get("total_final_cost"),
                })
            except Exception as e:
                logger.error(f"Failed to read invoice {path}: {e}")
        return invoices

    def get_invoice(self, invoice_id: str) -> Optional[Dict]:
        """Get a full invoice by ID."""
        path = self._invoice_path(invoice_id)
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)
