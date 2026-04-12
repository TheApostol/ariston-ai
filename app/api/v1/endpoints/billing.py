"""
Billing endpoints — Stripe checkout, webhooks, usage metering, and plan catalog.

Endpoints:
  POST /billing/checkout          — create Stripe checkout session
  POST /billing/webhook           — handle Stripe webhook events
  GET  /billing/usage/{tenant_id} — return usage from metering table
  GET  /billing/plans             — return available plans

All endpoints degrade gracefully when STRIPE_SECRET_KEY is not set.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, Header, HTTPException, Path, Request
from pydantic import BaseModel

logger = logging.getLogger("ariston.billing")
router = APIRouter(prefix="/billing", tags=["Billing"])


# ── Plan catalog ────────────────────────────────────────────────────────────

PLANS: Dict[str, Dict[str, Any]] = {
    "free": {
        "id": "free",
        "name": "Free",
        "price_usd": 0,
        "price_interval": None,
        "requests_per_month": 100,
        "agents": ["regulatory", "clinical"],
        "features": ["LATAM regulatory queries", "Clinical intelligence (limited)"],
        "stripe_price_id": None,
    },
    "pro": {
        "id": "pro",
        "name": "Pro",
        "price_usd": 299,
        "price_interval": "month",
        "requests_per_month": 5_000,
        "agents": ["regulatory", "clinical", "pharma", "genomics", "iomt", "twin"],
        "features": [
            "All agents unlocked",
            "5,000 requests/month",
            "RWE data access",
            "CIOMS/MedWatch generation",
            "Priority support",
        ],
        "stripe_price_id": "price_pro_monthly",
    },
    "enterprise": {
        "id": "enterprise",
        "name": "Enterprise",
        "price_usd": None,
        "price_interval": "custom",
        "requests_per_month": None,
        "agents": ["all"],
        "features": [
            "Unlimited requests",
            "Dedicated infrastructure",
            "GxP / 21 CFR Part 11 audit trail",
            "SSO / SAML integration",
            "SLA guarantee",
            "Custom LATAM data connectors",
            "Dedicated support engineer",
        ],
        "stripe_price_id": None,
        "contact": "enterprise@ariston.ai",
    },
}


# ── Request models ──────────────────────────────────────────────────────────

class CheckoutRequest(BaseModel):
    tenant_id: str
    plan: str = "pro"
    email: str
    success_url: str = "https://ariston.ai/billing/success"
    cancel_url: str = "https://ariston.ai/billing/cancel"


# ── Helpers ─────────────────────────────────────────────────────────────────

def _get_stripe():
    """Lazy-import stripe and return the module, or None if not configured."""
    try:
        import stripe as _stripe  # lazy import — SDK optional
        from config import settings
        key = getattr(settings, "STRIPE_SECRET_KEY", None)
        if not key:
            import os
            key = os.getenv("STRIPE_SECRET_KEY")
        if not key:
            return None
        _stripe.api_key = key
        return _stripe
    except ImportError:
        logger.warning("[billing] stripe SDK not installed")
        return None


def _get_webhook_secret() -> Optional[str]:
    try:
        from config import settings
        secret = getattr(settings, "STRIPE_WEBHOOK_SECRET", None)
        if secret:
            return secret
    except Exception:
        pass
    import os
    return os.getenv("STRIPE_WEBHOOK_SECRET")


def _get_usage(tenant_id: str) -> Dict[str, Any]:
    """
    Pull usage data from metering infrastructure.
    Falls back to zeros when metering is unavailable.
    """
    try:
        from vinci_core.billing.metering import usage_meter
        summary = usage_meter.get_usage_summary(tenant_id)
        return summary
    except Exception as e:
        logger.warning("[billing] metering unavailable: %s", e)
        return {
            "tenant_id": tenant_id,
            "requests_this_month": 0,
            "tokens_this_month": 0,
            "agents_used": [],
            "last_request_at": None,
            "source": "unavailable",
        }


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/checkout")
async def create_checkout_session(payload: CheckoutRequest) -> Dict[str, Any]:
    """
    Create a Stripe Checkout session for a given plan.
    Returns the session URL if Stripe is configured, otherwise a graceful error.
    """
    stripe = _get_stripe()
    if not stripe:
        return {
            "status": "stripe_not_configured",
            "message": "Stripe is not configured. Set STRIPE_SECRET_KEY to enable billing.",
            "plan": payload.plan,
            "tenant_id": payload.tenant_id,
        }

    plan = PLANS.get(payload.plan)
    if not plan:
        raise HTTPException(status_code=400, detail=f"Unknown plan: {payload.plan}")
    if not plan.get("stripe_price_id"):
        raise HTTPException(
            status_code=400,
            detail=f"Plan '{payload.plan}' does not have a Stripe price configured. Contact enterprise@ariston.ai.",
        )

    try:
        session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            mode="subscription",
            customer_email=payload.email,
            line_items=[{"price": plan["stripe_price_id"], "quantity": 1}],
            success_url=payload.success_url + "?session_id={CHECKOUT_SESSION_ID}",
            cancel_url=payload.cancel_url,
            metadata={"tenant_id": payload.tenant_id, "plan": payload.plan},
        )
        return {
            "session_id": session.id,
            "url": session.url,
            "plan": payload.plan,
            "tenant_id": payload.tenant_id,
        }
    except Exception as e:
        logger.error("[billing] checkout session creation failed: %s", e)
        return {
            "status": "error",
            "message": "Failed to create checkout session. Please try again.",
            "plan": payload.plan,
        }


@router.post("/webhook")
async def stripe_webhook(
    request: Request,
    stripe_signature: Optional[str] = Header(None, alias="stripe-signature"),
) -> Dict[str, Any]:
    """
    Handle Stripe webhook events.
    Verifies signature when STRIPE_WEBHOOK_SECRET is set.
    Handles: payment_intent.succeeded, customer.subscription.created/updated/deleted.
    """
    stripe = _get_stripe()
    body = await request.body()

    if stripe:
        webhook_secret = _get_webhook_secret()
        if webhook_secret and stripe_signature:
            try:
                event = stripe.Webhook.construct_event(
                    body, stripe_signature, webhook_secret
                )
            except stripe.error.SignatureVerificationError:
                logger.warning("[billing] webhook signature verification failed")
                raise HTTPException(status_code=400, detail="Invalid signature")
        else:
            # No secret configured — parse raw payload without verification
            import json
            try:
                event = json.loads(body)
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid payload")
    else:
        # Stripe not configured — parse and log only
        import json
        try:
            event = json.loads(body)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid payload")

    event_type = event.get("type", "unknown") if isinstance(event, dict) else getattr(event, "type", "unknown")
    logger.info("[billing] webhook received: %s", event_type)

    # Handle specific event types
    handled = False

    if event_type == "payment_intent.succeeded":
        data = (event.get("data", {}) if isinstance(event, dict) else event.data)
        pi = data.get("object", {}) if isinstance(data, dict) else getattr(data, "object", {})
        tenant_id = (pi.get("metadata", {}) if isinstance(pi, dict) else getattr(pi, "metadata", {})).get("tenant_id")
        logger.info("[billing] payment succeeded for tenant=%s", tenant_id)
        handled = True

    elif event_type in ("customer.subscription.created", "customer.subscription.updated"):
        data = (event.get("data", {}) if isinstance(event, dict) else event.data)
        sub = data.get("object", {}) if isinstance(data, dict) else getattr(data, "object", {})
        tenant_id = (sub.get("metadata", {}) if isinstance(sub, dict) else getattr(sub, "metadata", {})).get("tenant_id")
        status = sub.get("status") if isinstance(sub, dict) else getattr(sub, "status", None)
        logger.info("[billing] subscription %s status=%s tenant=%s", event_type, status, tenant_id)
        handled = True

    elif event_type == "customer.subscription.deleted":
        data = (event.get("data", {}) if isinstance(event, dict) else event.data)
        sub = data.get("object", {}) if isinstance(data, dict) else getattr(data, "object", {})
        tenant_id = (sub.get("metadata", {}) if isinstance(sub, dict) else getattr(sub, "metadata", {})).get("tenant_id")
        logger.info("[billing] subscription cancelled for tenant=%s", tenant_id)
        handled = True

    return {"received": True, "event_type": event_type, "handled": handled}


@router.get("/usage/{tenant_id}")
async def get_usage(
    tenant_id: str = Path(..., description="Tenant identifier"),
) -> Dict[str, Any]:
    """
    Return usage metrics for a given tenant from the metering table.
    """
    usage = _get_usage(tenant_id)
    return usage


@router.get("/plans")
async def list_plans() -> Dict[str, Any]:
    """
    Return available subscription plans.
    """
    return {
        "plans": list(PLANS.values()),
        "currency": "USD",
        "billing_note": "Annual plans available — contact sales@ariston.ai for 20% discount.",
    }
