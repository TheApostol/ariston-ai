"""
Webhook Event Dispatcher — Phase 4 / Ariston AI.

Outbound event notifications for trial milestones, regulatory alerts,
PV signal flags, and pipeline completions.

Enterprise pharma workflows require automated notifications when:
  - A clinical trial milestone is reached (enrollment complete, DSMB review)
  - A regulatory submission status changes (ANVISA approved, COFEPRIS query)
  - A pharmacovigilance signal is flagged (safety signal above threshold)
  - A drug discovery hypothesis crosses confidence threshold
  - RWE data freshness drops below SLA threshold

Architecture:
  - SQLite-backed subscription registry (tenant-scoped)
  - HMAC-SHA256 signed payloads (webhook consumers verify authenticity)
  - Retry with exponential backoff (3 attempts, 5s/25s/125s delays)
  - Dead letter queue for failed deliveries
  - Per-tenant event filtering (subscribe to specific event types only)
  - Async delivery (fire-and-forget, non-blocking main request path)
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Generator, Optional

logger = logging.getLogger("ariston.webhooks")

_DB_PATH = os.environ.get("ARISTON_WEBHOOK_DB", "data/webhooks.db")
_MAX_RETRIES = 3
_RETRY_DELAYS = [5, 25, 125]  # exponential backoff in seconds

# ---------------------------------------------------------------------------
# Event catalogue
# ---------------------------------------------------------------------------
WEBHOOK_EVENT_TYPES = [
    # Trial intelligence
    "trial.enrollment_complete",
    "trial.regulatory_approval",
    "trial.site_activated",
    "trial.protocol_amendment",
    "trial.dsmb_review_due",
    # Pharmacovigilance
    "pv.signal_flagged",
    "pv.case_serious",
    "pv.latam_report_due",
    # Drug discovery
    "drug_discovery.hypothesis_validated",
    "drug_discovery.hypothesis_rejected",
    "drug_discovery.candidate_advanced",
    # Regulatory
    "regulatory.submission_accepted",
    "regulatory.query_received",
    "regulatory.approval_granted",
    "regulatory.approval_refused",
    # Pipeline
    "pipeline.completed",
    "pipeline.failed",
    "pipeline.safety_block",
    # RWE
    "rwe.freshness_below_sla",
    "rwe.dataset_registered",
    "rwe.licensing_proposal_accepted",
    # System
    "system.audit_chain_broken",
    "system.rate_limit_approaching",
]


@dataclass
class WebhookEvent:
    """An event payload to be dispatched to subscribers."""
    event_id: str
    event_type: str
    tenant_id: str
    payload: dict
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source: str = "ariston-ai"


@dataclass
class WebhookSubscription:
    sub_id: str
    tenant_id: str
    url: str
    secret: str           # HMAC signing secret (stored, used to sign payloads)
    event_types: list[str]  # empty = subscribe to all
    active: bool = True
    created_at: str = ""
    description: str = ""


class WebhookDispatcher:
    """
    Tenant-scoped webhook subscription management and async delivery.

    Provides:
    - subscribe()     — register a webhook endpoint for a tenant
    - unsubscribe()   — remove a subscription
    - emit()          — fire an event (dispatched async to all matching subscribers)
    - get_deliveries()— query delivery history with status
    - retry_failed()  — manually retry dead-lettered deliveries
    """

    def __init__(self, db_path: str = _DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else "data", exist_ok=True)
        self._init_db()

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS subscriptions (
                    sub_id      TEXT PRIMARY KEY,
                    tenant_id   TEXT NOT NULL,
                    url         TEXT NOT NULL,
                    secret      TEXT NOT NULL,
                    event_types TEXT NOT NULL DEFAULT '[]',
                    active      INTEGER NOT NULL DEFAULT 1,
                    created_at  TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS deliveries (
                    delivery_id TEXT PRIMARY KEY,
                    event_id    TEXT NOT NULL,
                    sub_id      TEXT NOT NULL,
                    tenant_id   TEXT NOT NULL,
                    event_type  TEXT NOT NULL,
                    url         TEXT NOT NULL,
                    status      TEXT NOT NULL DEFAULT 'pending',
                    attempts    INTEGER NOT NULL DEFAULT 0,
                    last_attempt_at TEXT,
                    response_code   INTEGER,
                    error_message   TEXT,
                    payload_json    TEXT,
                    created_at  TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_del_event  ON deliveries(event_id);
                CREATE INDEX IF NOT EXISTS idx_del_tenant ON deliveries(tenant_id);
                CREATE INDEX IF NOT EXISTS idx_del_status ON deliveries(status);
            """)

    def subscribe(
        self,
        url: str,
        tenant_id: str,
        event_types: Optional[list[str]] = None,
        description: str = "",
        secret: Optional[str] = None,
    ) -> WebhookSubscription:
        """Register a webhook endpoint for a tenant."""
        import secrets as sec
        # Validate event types
        if event_types:
            invalid = [e for e in event_types if e not in WEBHOOK_EVENT_TYPES]
            if invalid:
                raise ValueError(f"Unknown event types: {invalid}. Valid: {WEBHOOK_EVENT_TYPES}")

        sub = WebhookSubscription(
            sub_id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            url=url,
            secret=secret or sec.token_hex(32),
            event_types=event_types or [],
            created_at=datetime.now(timezone.utc).isoformat(),
            description=description,
        )
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO subscriptions
                  (sub_id, tenant_id, url, secret, event_types, created_at, description)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                sub.sub_id, sub.tenant_id, sub.url, sub.secret,
                json.dumps(sub.event_types), sub.created_at, sub.description,
            ))
        logger.info("[Webhook] subscribed sub_id=%s tenant=%s url=%s", sub.sub_id, tenant_id, url)
        return sub

    def unsubscribe(self, sub_id: str, tenant_id: str) -> bool:
        with self._conn() as conn:
            r = conn.execute(
                "UPDATE subscriptions SET active = 0 WHERE sub_id = ? AND tenant_id = ?",
                (sub_id, tenant_id),
            )
        return r.rowcount > 0

    def emit(self, event: WebhookEvent, background: bool = True) -> list[str]:
        """
        Emit an event — dispatches to all matching subscribers.
        Returns list of delivery_ids created.

        background=True: schedules async delivery (non-blocking)
        background=False: delivers synchronously (for testing)
        """
        subscribers = self._matching_subscribers(event.tenant_id, event.event_type)
        if not subscribers:
            return []

        delivery_ids = []
        for sub in subscribers:
            delivery_id = str(uuid.uuid4())
            payload_json = json.dumps({
                "event_id": event.event_id,
                "event_type": event.event_type,
                "tenant_id": event.tenant_id,
                "source": event.source,
                "created_at": event.created_at,
                "payload": event.payload,
            }, default=str)

            with self._conn() as conn:
                conn.execute("""
                    INSERT INTO deliveries
                      (delivery_id, event_id, sub_id, tenant_id, event_type,
                       url, status, payload_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?)
                """, (
                    delivery_id, event.event_id, sub.sub_id, event.tenant_id,
                    event.event_type, sub.url, payload_json,
                    datetime.now(timezone.utc).isoformat(),
                ))
            delivery_ids.append(delivery_id)

            if background:
                asyncio.create_task(self._deliver(delivery_id, sub, payload_json))
            else:
                # Synchronous path for testing — create a new loop if needed
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_closed():
                        raise RuntimeError("closed")
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                loop.run_until_complete(self._deliver(delivery_id, sub, payload_json))

        return delivery_ids

    def get_deliveries(
        self,
        tenant_id: str,
        event_type: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict]:
        clauses = ["tenant_id = ?"]
        params: list[Any] = [tenant_id]
        if event_type:
            clauses.append("event_type = ?")
            params.append(event_type)
        if status:
            clauses.append("status = ?")
            params.append(status)
        params += [limit]

        with self._conn() as conn:
            rows = conn.execute(f"""
                SELECT delivery_id, event_id, event_type, url, status, attempts,
                       last_attempt_at, response_code, error_message, created_at
                FROM deliveries
                WHERE {' AND '.join(clauses)}
                ORDER BY created_at DESC LIMIT ?
            """, params).fetchall()
        return [dict(r) for r in rows]

    def get_subscriptions(self, tenant_id: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT sub_id, url, event_types, active, created_at, description FROM subscriptions WHERE tenant_id = ? ORDER BY created_at DESC",
                (tenant_id,),
            ).fetchall()
        return [
            {
                "sub_id": r["sub_id"],
                "url": r["url"],
                "event_types": json.loads(r["event_types"] or "[]"),
                "active": bool(r["active"]),
                "created_at": r["created_at"],
                "description": r["description"],
            }
            for r in rows
        ]

    # ---------------------------------------------------------------------------
    # Delivery logic
    # ---------------------------------------------------------------------------

    async def _deliver(self, delivery_id: str, sub: WebhookSubscription, payload_json: str) -> None:
        import httpx
        signature = hmac.new(
            sub.secret.encode(), payload_json.encode(), hashlib.sha256
        ).hexdigest()

        headers = {
            "Content-Type": "application/json",
            "X-Ariston-Signature": f"sha256={signature}",
            "X-Ariston-Event": delivery_id,
            "User-Agent": "AristonAI-Webhook/1.0",
        }

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    resp = await client.post(sub.url, content=payload_json, headers=headers)

                status_str = "delivered" if resp.status_code < 400 else "failed"
                with self._conn() as conn:
                    conn.execute("""
                        UPDATE deliveries SET status=?, attempts=?, last_attempt_at=?, response_code=?
                        WHERE delivery_id=?
                    """, (status_str, attempt, datetime.now(timezone.utc).isoformat(), resp.status_code, delivery_id))

                if resp.status_code < 400:
                    logger.info("[Webhook] delivered delivery_id=%s attempt=%d", delivery_id, attempt)
                    return

            except Exception as e:
                err = str(e)
                with self._conn() as conn:
                    conn.execute("""
                        UPDATE deliveries SET status='failed', attempts=?, last_attempt_at=?, error_message=?
                        WHERE delivery_id=?
                    """, (attempt, datetime.now(timezone.utc).isoformat(), err[:500], delivery_id))

                if attempt < _MAX_RETRIES:
                    await asyncio.sleep(_RETRY_DELAYS[attempt - 1])

        # All retries exhausted → dead letter
        with self._conn() as conn:
            conn.execute(
                "UPDATE deliveries SET status='dead_letter' WHERE delivery_id=?",
                (delivery_id,),
            )
        logger.error("[Webhook] dead_letter delivery_id=%s sub=%s", delivery_id, sub.sub_id)

    def _matching_subscribers(self, tenant_id: str, event_type: str) -> list[WebhookSubscription]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM subscriptions WHERE tenant_id = ? AND active = 1",
                (tenant_id,),
            ).fetchall()

        subs = []
        for r in rows:
            event_types = json.loads(r["event_types"] or "[]")
            if not event_types or event_type in event_types:
                subs.append(WebhookSubscription(
                    sub_id=r["sub_id"],
                    tenant_id=r["tenant_id"],
                    url=r["url"],
                    secret=r["secret"],
                    event_types=event_types,
                    active=bool(r["active"]),
                    created_at=r["created_at"],
                    description=r["description"],
                ))
        return subs


webhook_dispatcher = WebhookDispatcher()
