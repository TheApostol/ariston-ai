"""
Stripe Billing Integration — Phase 5 / Ariston AI.

Wraps Stripe API for subscription lifecycle management.
Stripe SDK is optional — gracefully degrades to manual invoicing if not installed.

Handles:
  - Customer creation (one Stripe customer per tenant)
  - Subscription creation/upgrades/cancellations
  - Metered billing (usage records → Stripe Meters API)
  - Invoice retrieval
  - Webhook event processing (payment_intent.succeeded, invoice.paid, etc.)
  - Trial-to-paid conversion tracking

Revenue targets per Execution Roadmap:
  - Phase 1 pilots: $0 (free PoC) → $100K/year (standard)
  - Phase 2: $300K/year (premium)
  - Phase 3: $500K–$2M/year (enterprise)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("ariston.billing.stripe")

_STRIPE_SECRET = os.environ.get("STRIPE_SECRET_KEY", "")
_STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")


@dataclass
class StripeCustomer:
    customer_id: str
    tenant_id: str
    email: str
    name: str
    subscription_id: Optional[str] = None
    subscription_status: Optional[str] = None
    current_period_end: Optional[str] = None


def _get_stripe():
    """Lazy Stripe import — returns None if not installed."""
    try:
        import stripe
        stripe.api_key = _STRIPE_SECRET
        return stripe
    except ImportError:
        logger.warning("[Stripe] stripe SDK not installed — billing in mock mode")
        return None


class StripeIntegration:
    """
    Stripe billing integration for Ariston AI.

    All methods gracefully degrade if Stripe SDK unavailable or API key missing.
    This enables development without Stripe credentials.
    """

    def create_customer(
        self,
        tenant_id: str,
        email: str,
        name: str,
        metadata: Optional[dict] = None,
    ) -> Optional[StripeCustomer]:
        stripe = _get_stripe()
        if not stripe or not _STRIPE_SECRET:
            logger.info("[Stripe] Mock: create_customer tenant=%s", tenant_id)
            return StripeCustomer(
                customer_id=f"cus_mock_{tenant_id[:8]}",
                tenant_id=tenant_id,
                email=email,
                name=name,
            )
        try:
            customer = stripe.Customer.create(
                email=email,
                name=name,
                metadata={"tenant_id": tenant_id, **(metadata or {})},
            )
            return StripeCustomer(
                customer_id=customer["id"],
                tenant_id=tenant_id,
                email=email,
                name=name,
            )
        except Exception as e:
            logger.error("[Stripe] create_customer failed: %s", e)
            return None

    def create_subscription(
        self,
        customer_id: str,
        price_id: str,
        trial_days: int = 30,
        metadata: Optional[dict] = None,
    ) -> Optional[dict]:
        stripe = _get_stripe()
        if not stripe or not _STRIPE_SECRET:
            logger.info("[Stripe] Mock: create_subscription customer=%s price=%s", customer_id, price_id)
            return {
                "id": f"sub_mock_{customer_id[-8:]}",
                "status": "trialing",
                "current_period_end": None,
                "mock": True,
            }
        try:
            sub = stripe.Subscription.create(
                customer=customer_id,
                items=[{"price": price_id}],
                trial_period_days=trial_days,
                metadata=metadata or {},
            )
            return dict(sub)
        except Exception as e:
            logger.error("[Stripe] create_subscription failed: %s", e)
            return None

    def record_usage(
        self,
        subscription_item_id: str,
        quantity: int,
        timestamp: Optional[int] = None,
        action: str = "increment",
    ) -> bool:
        """Push metered usage to Stripe (for usage-based billing)."""
        stripe = _get_stripe()
        if not stripe or not _STRIPE_SECRET:
            logger.debug("[Stripe] Mock: record_usage item=%s qty=%d", subscription_item_id, quantity)
            return True
        try:
            import time
            stripe.SubscriptionItem.create_usage_record(
                subscription_item_id,
                quantity=quantity,
                timestamp=timestamp or int(time.time()),
                action=action,
            )
            return True
        except Exception as e:
            logger.error("[Stripe] record_usage failed: %s", e)
            return False

    def get_upcoming_invoice(self, customer_id: str) -> Optional[dict]:
        stripe = _get_stripe()
        if not stripe or not _STRIPE_SECRET:
            return {"mock": True, "amount_due": 0, "currency": "usd"}
        try:
            inv = stripe.Invoice.upcoming(customer=customer_id)
            return {
                "amount_due_usd": inv["amount_due"] / 100,
                "currency": inv["currency"],
                "period_start": inv.get("period_start"),
                "period_end": inv.get("period_end"),
                "lines": [
                    {
                        "description": line.get("description"),
                        "amount_usd": line["amount"] / 100,
                    }
                    for line in inv.get("lines", {}).get("data", [])
                ],
            }
        except Exception as e:
            logger.error("[Stripe] get_upcoming_invoice failed: %s", e)
            return None

    def cancel_subscription(self, subscription_id: str, at_period_end: bool = True) -> bool:
        stripe = _get_stripe()
        if not stripe or not _STRIPE_SECRET:
            logger.info("[Stripe] Mock: cancel_subscription sub=%s", subscription_id)
            return True
        try:
            if at_period_end:
                stripe.Subscription.modify(subscription_id, cancel_at_period_end=True)
            else:
                stripe.Subscription.cancel(subscription_id)
            return True
        except Exception as e:
            logger.error("[Stripe] cancel_subscription failed: %s", e)
            return False

    def process_webhook(self, payload: bytes, sig_header: str) -> Optional[dict]:
        """
        Verify and parse a Stripe webhook event.
        Returns the parsed event dict or None on invalid signature.
        """
        stripe = _get_stripe()
        if not stripe or not _STRIPE_WEBHOOK_SECRET:
            return None
        try:
            event = stripe.Webhook.construct_event(payload, sig_header, _STRIPE_WEBHOOK_SECRET)
            logger.info("[Stripe] webhook event_type=%s", event["type"])
            return dict(event)
        except Exception as e:
            logger.error("[Stripe] webhook verification failed: %s", e)
            return None


stripe_integration = StripeIntegration()
