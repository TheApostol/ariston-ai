"""
Phase 5 API — Revenue Infrastructure / Ariston AI.

Customer portal, billing dashboard, SLA reporting, Stripe webhook handler.

Endpoints:
  GET  /phase5/billing/plans              — list subscription plans
  GET  /phase5/billing/usage              — current usage + quota status
  GET  /phase5/billing/invoice            — current period invoice
  POST /phase5/billing/subscribe          — create/upgrade subscription
  GET  /phase5/sla/report                 — SLA performance report
  GET  /phase5/sla/uptime                 — daily uptime series
  POST /phase5/sla/record                 — record request metric (internal)
  POST /phase5/stripe/webhook             — Stripe webhook processor
  GET  /phase5/portal/dashboard           — customer dashboard summary
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

logger = logging.getLogger("ariston.phase5")
router = APIRouter(prefix="/phase5", tags=["Phase 5 — Revenue Infrastructure"])


# ── Request/response models ─────────────────────────────────────────────────

class SubscribeRequest(BaseModel):
    tenant_id: str
    tier: str                          # pilot | standard | premium | enterprise
    email: str
    company_name: str
    trial_days: int = 30


class RecordMetricRequest(BaseModel):
    latency_ms: float
    success: bool = True
    tenant_id: str = "global"
    endpoint: str = "unknown"
    layer: str = "unknown"
    error_type: Optional[str] = None
    rag_used: bool = False
    rag_latency_ms: float = 0.0


# ── Billing endpoints ────────────────────────────────────────────────────────

@router.get("/billing/plans")
async def list_plans():
    """All subscription plans with pricing and features."""
    from vinci_core.billing.plans import PLANS
    return {
        "plans": {
            tier: {
                "name": p["name"],
                "description": p["description"],
                "price_usd_year": p["price_usd_year"],
                "price_usd_month": p["price_usd_month"],
                "features": p["features"],
                "included_units": {
                    k: v for k, v in p["included_units"].items()
                    if k in ("api_calls", "pipeline_runs", "rag_queries")
                },
            }
            for tier, p in PLANS.items()
        }
    }


@router.get("/billing/usage")
async def get_usage(
    tenant_id: str = Query("demo_tenant"),
    tier: str = Query("standard"),
):
    """Current period usage vs quota for a tenant."""
    from vinci_core.billing.metering import usage_meter
    summary = usage_meter.get_summary(tenant_id=tenant_id, tier=tier)
    quota_status = {}
    for unit in ["api_calls", "pipeline_runs", "rag_queries"]:
        quota_status[unit] = usage_meter.check_quota(tenant_id, unit, tier)
    return {
        "tenant_id": summary.tenant_id,
        "tier": summary.tier,
        "period_start": summary.period_start,
        "period_end": summary.period_end,
        "usage": summary.units,
        "quota": summary.quota,
        "overage_usd": summary.overage,
        "total_overage_usd": summary.total_overage_usd,
        "within_quota": summary.within_quota,
        "quota_status": quota_status,
    }


@router.get("/billing/invoice")
async def get_invoice(
    tenant_id: str = Query("demo_tenant"),
    tier: str = Query("standard"),
):
    """Invoice line items for current billing period."""
    from vinci_core.billing.metering import usage_meter
    return usage_meter.get_invoice_data(tenant_id=tenant_id, tier=tier)


@router.post("/billing/subscribe")
async def subscribe(req: SubscribeRequest):
    """Create a new subscription (Stripe customer + subscription)."""
    from vinci_core.billing.stripe_integration import stripe_integration
    from vinci_core.billing.plans import PLANS

    plan = PLANS.get(req.tier.lower())
    if not plan:
        raise HTTPException(status_code=400, detail=f"Unknown tier: {req.tier}")

    customer = stripe_integration.create_customer(
        tenant_id=req.tenant_id,
        email=req.email,
        name=req.company_name,
        metadata={"tier": req.tier},
    )
    if not customer:
        raise HTTPException(status_code=500, detail="Failed to create Stripe customer")

    subscription = None
    if plan["stripe_price_id"]:
        subscription = stripe_integration.create_subscription(
            customer_id=customer.customer_id,
            price_id=plan["stripe_price_id"],
            trial_days=req.trial_days,
        )

    return {
        "tenant_id": req.tenant_id,
        "tier": req.tier,
        "customer_id": customer.customer_id,
        "subscription": subscription,
        "trial_days": req.trial_days,
        "plan": {
            "name": plan["name"],
            "price_usd_year": plan["price_usd_year"],
            "features": plan["features"],
        },
    }


@router.get("/billing/usage/daily")
async def get_daily_usage(
    tenant_id: str = Query("demo_tenant"),
    unit: str = Query("api_calls"),
    days: int = Query(30),
):
    """Daily usage breakdown for dashboard chart."""
    from vinci_core.billing.metering import usage_meter
    return {
        "tenant_id": tenant_id,
        "unit": unit,
        "days": days,
        "breakdown": usage_meter.get_daily_breakdown(tenant_id, unit, days),
    }


# ── SLA endpoints ────────────────────────────────────────────────────────────

@router.get("/sla/report")
async def get_sla_report(
    tenant_id: str = Query("global"),
    tier: str = Query("standard"),
    window_hours: int = Query(24),
):
    """SLA performance report: latency percentiles, uptime, breaches."""
    from vinci_core.sla.monitor import sla_monitor
    report = sla_monitor.get_report(
        tenant_id=tenant_id,
        tier=tier,
        window_hours=window_hours,
    )
    return {
        "tenant_id": report.tenant_id,
        "tier": report.tier,
        "window_hours": report.window_hours,
        "period_start": report.period_start,
        "period_end": report.period_end,
        "metrics": {
            "total_requests": report.total_requests,
            "successful_requests": report.successful_requests,
            "error_rate_pct": report.error_rate_pct,
            "p50_latency_ms": report.p50_latency_ms,
            "p95_latency_ms": report.p95_latency_ms,
            "p99_latency_ms": report.p99_latency_ms,
            "uptime_pct": report.uptime_pct,
        },
        "sla_targets": report.sla_targets,
        "breaches": report.breaches,
        "sla_status": "breach" if report.breaches else "healthy",
    }


@router.get("/sla/uptime")
async def get_uptime_series(
    tenant_id: str = Query("global"),
    days: int = Query(30),
):
    """Daily uptime percentages for dashboard chart."""
    from vinci_core.sla.monitor import sla_monitor
    return {
        "tenant_id": tenant_id,
        "days": days,
        "series": sla_monitor.get_uptime_series(tenant_id=tenant_id, days=days),
    }


@router.post("/sla/record")
async def record_metric(req: RecordMetricRequest):
    """Record a request metric into SLA store (called internally by engine)."""
    from vinci_core.sla.monitor import sla_monitor
    metric = sla_monitor.record(
        latency_ms=req.latency_ms,
        success=req.success,
        tenant_id=req.tenant_id,
        endpoint=req.endpoint,
        layer=req.layer,
        error_type=req.error_type,
        rag_used=req.rag_used,
        rag_latency_ms=req.rag_latency_ms,
    )
    return {
        "metric_id": metric.metric_id,
        "recorded": True,
        "latency_ms": metric.latency_ms,
        "success": metric.success,
    }


# ── Stripe webhook ───────────────────────────────────────────────────────────

@router.post("/stripe/webhook")
async def stripe_webhook(request: Request):
    """
    Receive and process Stripe webhook events.
    Verifies HMAC signature before processing.
    """
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature", "")

    from vinci_core.billing.stripe_integration import stripe_integration
    event = stripe_integration.process_webhook(payload, sig_header)

    if event is None:
        # No webhook secret configured — log and accept in dev mode
        logger.warning("[Stripe] Webhook received but not verified (no STRIPE_WEBHOOK_SECRET)")
        return {"received": True, "verified": False}

    event_type = event.get("type", "unknown")
    logger.info("[Stripe] Processed webhook event_type=%s", event_type)

    # Handle key billing lifecycle events
    if event_type == "invoice.paid":
        logger.info("[Stripe] Invoice paid: %s", event.get("id"))
    elif event_type == "customer.subscription.deleted":
        logger.info("[Stripe] Subscription cancelled: %s", event.get("id"))
    elif event_type == "payment_intent.payment_failed":
        logger.warning("[Stripe] Payment failed: %s", event.get("id"))

    return {"received": True, "verified": True, "event_type": event_type}


# ── Customer portal dashboard ────────────────────────────────────────────────

@router.get("/portal/dashboard")
async def get_portal_dashboard(
    tenant_id: str = Query("demo_tenant"),
    tier: str = Query("standard"),
):
    """
    Full customer portal dashboard: usage + SLA + billing summary.
    Single endpoint for the frontend dashboard widget.
    """
    from vinci_core.billing.metering import usage_meter
    from vinci_core.sla.monitor import sla_monitor
    from vinci_core.billing.plans import get_plan

    plan = get_plan(tier)
    summary = usage_meter.get_summary(tenant_id=tenant_id, tier=tier)
    sla_report = sla_monitor.get_report(tenant_id=tenant_id, tier=tier, window_hours=24)
    invoice = usage_meter.get_invoice_data(tenant_id=tenant_id, tier=tier)

    # Usage utilization percentages
    utilization = {}
    for unit in ["api_calls", "pipeline_runs", "rag_queries"]:
        limit = plan["included_units"].get(unit, 1)
        used = summary.units.get(unit, 0)
        pct = round(used / limit * 100, 1) if limit > 0 and limit != -1 else 0.0
        utilization[unit] = {"used": used, "limit": limit, "pct": pct}

    return {
        "tenant_id": tenant_id,
        "tier": tier,
        "plan_name": plan["name"],
        "billing": {
            "period_start": summary.period_start,
            "period_end": summary.period_end,
            "base_usd": invoice["base_subscription_usd"],
            "overage_usd": summary.total_overage_usd,
            "total_due_usd": invoice["total_due_usd"],
        },
        "utilization": utilization,
        "sla": {
            "uptime_pct": sla_report.uptime_pct,
            "p50_ms": sla_report.p50_latency_ms,
            "p95_ms": sla_report.p95_latency_ms,
            "status": "breach" if sla_report.breaches else "healthy",
            "breaches": sla_report.breaches,
        },
        "features": plan["features"],
    }
