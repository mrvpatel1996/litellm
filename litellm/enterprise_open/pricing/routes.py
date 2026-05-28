"""
Pricing & Billing API routes.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request

from litellm.enterprise_open.config import PricingRuleConfig
from litellm.enterprise_open.pricing.billing import InvoiceService
from litellm.enterprise_open.pricing.engine import get_pricing_engine

logger = logging.getLogger(__name__)
pricing_router = APIRouter()


# ── Pricing Rules ────────────────────────────────────────

@pricing_router.get("/rules")
async def list_pricing_rules(request: Request):
    """List all pricing rules."""
    engine = get_pricing_engine()
    return {"rules": engine.list_rules(), "count": len(engine.list_rules())}


@pricing_router.post("/rules")
async def create_pricing_rule(rule: PricingRuleConfig, request: Request):
    """Create a new pricing rule."""
    engine = get_pricing_engine()
    existing = engine.get_rule(rule.name)
    if existing:
        raise HTTPException(status_code=409, detail=f"Rule '{rule.name}' already exists")
    created = engine.add_rule(rule)
    return {"status": "ok", "rule": created.model_dump()}


@pricing_router.put("/rules/{rule_name}")
async def update_pricing_rule(rule_name: str, rule: PricingRuleConfig, request: Request):
    """Update a pricing rule."""
    engine = get_pricing_engine()
    updated = engine.update_rule(
        rule_name,
        model_pattern=rule.model_pattern,
        markup_percent=rule.markup_percent,
        flat_fee_per_request=rule.flat_fee_per_request,
        custom_price_input_per_1k=rule.custom_price_input_per_1k,
        custom_price_output_per_1k=rule.custom_price_output_per_1k,
        priority=rule.priority,
        description=rule.description,
    )
    if not updated:
        raise HTTPException(status_code=404, detail=f"Rule '{rule_name}' not found")
    return {"status": "ok", "rule": updated.model_dump()}


@pricing_router.delete("/rules/{rule_name}")
async def delete_pricing_rule(rule_name: str, request: Request):
    """Delete a pricing rule."""
    engine = get_pricing_engine()
    deleted = engine.delete_rule(rule_name)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Rule '{rule_name}' not found")
    return {"status": "ok", "message": f"Rule '{rule_name}' deleted"}


# ── Cost Calculator ──────────────────────────────────────

@pricing_router.post("/calculate")
async def calculate_cost(request: Request):
    """
    Preview cost calculation for a request.

    Body JSON:
    {
        "model": "gpt-4",
        "base_cost": 0.05,
        "input_tokens": 1000,
        "output_tokens": 500,
        "organization_id": null,
        "team_id": null
    }
    """
    body = await request.json()
    engine = get_pricing_engine()
    result = engine.calculate_cost(
        model=body.get("model", ""),
        base_cost=float(body.get("base_cost", 0)),
        input_tokens=int(body.get("input_tokens", 0)),
        output_tokens=int(body.get("output_tokens", 0)),
        organization_id=body.get("organization_id"),
        team_id=body.get("team_id"),
    )
    return result


# ── Invoices ─────────────────────────────────────────────

@pricing_router.post("/invoices")
async def generate_invoice(request: Request):
    """
    Generate an invoice for a billing period.

    Body JSON:
    {
        "organization_id": "org-123",
        "team_id": null,
        "period_start": "2026-01-01T00:00:00Z",
        "period_end": "2026-01-31T23:59:59Z"
    }
    """
    body = await request.json()
    service = InvoiceService()
    try:
        invoice = await service.generate_invoice(
            organization_id=body.get("organization_id"),
            team_id=body.get("team_id"),
            period_start=body.get("period_start"),
            period_end=body.get("period_end"),
        )
        return invoice
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


@pricing_router.get("/invoices")
async def list_invoices(request: Request):
    """List all invoices."""
    service = InvoiceService()
    invoices = service.list_invoices()
    return {"invoices": invoices, "count": len(invoices)}


@pricing_router.get("/invoices/{invoice_id}")
async def get_invoice(invoice_id: str, request: Request):
    """Get a full invoice by ID."""
    service = InvoiceService()
    invoice = service.get_invoice(invoice_id)
    if not invoice:
        raise HTTPException(status_code=404, detail=f"Invoice '{invoice_id}' not found")
    return invoice
