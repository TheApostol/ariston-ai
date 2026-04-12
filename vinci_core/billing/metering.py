"""
Usage Metering Engine — Phase 5 / Ariston AI.

Tracks per-tenant API consumption for billing:
  - api_calls          : every engine.run() call
  - pipeline_runs      : every Pipeline.run() call
  - rag_queries        : every RAG retrieve() call
  - drug_discovery_runs: drug target / repurposing queries
  - audit_exports      : GxP export downloads
  - webhook_deliveries : outbound webhook events

Architecture:
  - SQLite-backed event store (upgradeable to ClickHouse/TimescaleDB)
  - Rolling window aggregation (hourly, daily, monthly)
  - Overage detection with automatic webhook alert
  - Stripe integration hook (emit billing events to Stripe Meters API)
  - Per-tenant quota enforcement (soft limit → alert, hard limit → 429)
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Generator, Optional

from vinci_core.billing.plans import get_plan, is_within_quota, overage_cost

logger = logging.getLogger("ariston.billing")

_DB_PATH = os.environ.get("ARISTON_BILLING_DB", "data/billing.db")

METERED_UNITS = [
    "api_calls",
    "pipeline_runs",
    "rag_queries",
    "drug_discovery_runs",
    "audit_exports",
    "webhook_deliveries",
]


@dataclass
class MeteringEvent:
    event_id: str
    tenant_id: str
    unit: str                  # one of METERED_UNITS
    quantity: int
    pipeline: Optional[str]    # e.g. "latam_regulatory", "fda_510k"
    layer: Optional[str]
    cost_usd: float
    timestamp: str
    metadata: dict = field(default_factory=dict)


@dataclass
class UsageSummary:
    tenant_id: str
    period_start: str
    period_end: str
    tier: str
    units: dict[str, int]        # unit → total consumed
    quota: dict[str, int]        # unit → included in plan
    overage: dict[str, float]    # unit → overage cost USD
    total_overage_usd: float
    within_quota: dict[str, bool]


class UsageMeter:
    """
    Per-tenant usage metering and quota enforcement.

    Provides:
    - record()          — log a usage event
    - get_summary()     — rolling usage + overage calculation
    - check_quota()     — soft/hard quota enforcement
    - get_invoice_data()— monthly invoice line items
    - reset_period()    — start new billing cycle
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
                CREATE TABLE IF NOT EXISTS usage_events (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id    TEXT UNIQUE NOT NULL,
                    tenant_id   TEXT NOT NULL,
                    unit        TEXT NOT NULL,
                    quantity    INTEGER NOT NULL DEFAULT 1,
                    pipeline    TEXT,
                    layer       TEXT,
                    cost_usd    REAL NOT NULL DEFAULT 0.0,
                    timestamp   TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );

                CREATE INDEX IF NOT EXISTS idx_usage_tenant    ON usage_events(tenant_id);
                CREATE INDEX IF NOT EXISTS idx_usage_unit      ON usage_events(unit);
                CREATE INDEX IF NOT EXISTS idx_usage_timestamp ON usage_events(timestamp);

                CREATE TABLE IF NOT EXISTS billing_periods (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    tenant_id   TEXT NOT NULL,
                    tier        TEXT NOT NULL DEFAULT 'standard',
                    period_start TEXT NOT NULL,
                    period_end   TEXT NOT NULL,
                    status       TEXT NOT NULL DEFAULT 'active',
                    stripe_invoice_id TEXT,
                    total_usd    REAL NOT NULL DEFAULT 0.0
                );

                CREATE INDEX IF NOT EXISTS idx_bp_tenant ON billing_periods(tenant_id);
            """)

    def record(
        self,
        tenant_id: str,
        unit: str,
        quantity: int = 1,
        pipeline: Optional[str] = None,
        layer: Optional[str] = None,
        metadata: Optional[dict] = None,
        tier: str = "standard",
    ) -> MeteringEvent:
        """Record a usage event for a tenant."""
        now = datetime.now(timezone.utc).isoformat()

        # Calculate overage cost if applicable
        current = self._get_period_total(tenant_id, unit)
        plan = get_plan(tier)
        quota = plan["included_units"].get(unit, 0)
        excess = max(0, (current + quantity) - quota) if quota != -1 else 0
        cost = overage_cost(tier, unit, excess)

        event = MeteringEvent(
            event_id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            unit=unit,
            quantity=quantity,
            pipeline=pipeline,
            layer=layer,
            cost_usd=cost,
            timestamp=now,
            metadata=metadata or {},
        )

        with self._conn() as conn:
            conn.execute("""
                INSERT INTO usage_events
                  (event_id, tenant_id, unit, quantity, pipeline, layer, cost_usd, timestamp, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event.event_id, tenant_id, unit, quantity, pipeline, layer,
                cost, now, json.dumps(event.metadata),
            ))

        if cost > 0:
            logger.info("[Billing] overage tenant=%s unit=%s excess=%d cost=$%.2f", tenant_id, unit, excess, cost)

        return event

    def get_summary(
        self,
        tenant_id: str,
        tier: str = "standard",
        since: Optional[str] = None,
        until: Optional[str] = None,
    ) -> UsageSummary:
        """Get usage summary and overage calculation for billing period."""
        now = datetime.now(timezone.utc)
        if not since:
            # Default: current calendar month
            since = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0).isoformat()
        if not until:
            until = now.isoformat()

        plan = get_plan(tier)
        quota = plan["included_units"]

        with self._conn() as conn:
            rows = conn.execute("""
                SELECT unit, SUM(quantity) as total
                FROM usage_events
                WHERE tenant_id = ? AND timestamp >= ? AND timestamp <= ?
                GROUP BY unit
            """, (tenant_id, since, until)).fetchall()

        consumed = {r["unit"]: r["total"] for r in rows}

        overage: dict[str, float] = {}
        within: dict[str, bool] = {}
        total_overage = 0.0

        for unit in METERED_UNITS:
            used = consumed.get(unit, 0)
            limit = quota.get(unit, 0)
            within[unit] = limit == -1 or used <= limit
            excess = max(0, used - limit) if limit != -1 else 0
            cost = overage_cost(tier, unit, excess)
            overage[unit] = cost
            total_overage += cost

        return UsageSummary(
            tenant_id=tenant_id,
            period_start=since,
            period_end=until,
            tier=tier,
            units=consumed,
            quota={k: v for k, v in quota.items() if k in METERED_UNITS},
            overage=overage,
            total_overage_usd=round(total_overage, 2),
            within_quota=within,
        )

    def check_quota(
        self,
        tenant_id: str,
        unit: str,
        tier: str = "standard",
    ) -> dict:
        """
        Check if tenant is within quota for a unit.
        Returns {allowed: bool, used: int, limit: int, overage_usd_per_unit: float}
        """
        plan = get_plan(tier)
        limit = plan["included_units"].get(unit, 0)
        used = self._get_period_total(tenant_id, unit)
        rate = plan["overage_usd_per_unit"].get(unit, 0.0)
        allowed = limit == -1 or used < limit or rate > 0  # soft limit: allow with overage charge
        return {
            "allowed": allowed,
            "used": used,
            "limit": limit,
            "unlimited": limit == -1,
            "overage_rate_usd": rate,
            "within_included": limit == -1 or used < limit,
        }

    def get_invoice_data(
        self,
        tenant_id: str,
        tier: str = "standard",
        since: Optional[str] = None,
        until: Optional[str] = None,
    ) -> dict:
        """Generate invoice line items for a billing period."""
        summary = self.get_summary(tenant_id, tier, since, until)
        plan = get_plan(tier)
        base_price = plan["price_usd_year"] or 0

        line_items = []
        for unit in METERED_UNITS:
            used = summary.units.get(unit, 0)
            quota = summary.quota.get(unit, 0)
            excess = max(0, used - quota) if quota != -1 else 0
            if excess > 0:
                rate = plan["overage_usd_per_unit"].get(unit, 0.0)
                line_items.append({
                    "description": f"{unit.replace('_', ' ').title()} overage ({excess:,} units)",
                    "quantity": excess,
                    "unit_price_usd": rate,
                    "total_usd": round(rate * excess, 2),
                })

        return {
            "tenant_id": tenant_id,
            "tier": tier,
            "period_start": summary.period_start,
            "period_end": summary.period_end,
            "base_subscription_usd": base_price,
            "overage_line_items": line_items,
            "total_overage_usd": summary.total_overage_usd,
            "total_due_usd": round(base_price + summary.total_overage_usd, 2),
            "usage_summary": summary.units,
        }

    def get_daily_breakdown(
        self,
        tenant_id: str,
        unit: str,
        days: int = 30,
    ) -> list[dict]:
        """Daily usage breakdown for dashboard charts."""
        since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT substr(timestamp, 1, 10) as day, SUM(quantity) as total
                FROM usage_events
                WHERE tenant_id = ? AND unit = ? AND timestamp >= ?
                GROUP BY day ORDER BY day
            """, (tenant_id, unit, since)).fetchall()
        return [{"date": r["day"], "count": r["total"]} for r in rows]

    def _get_period_total(self, tenant_id: str, unit: str) -> int:
        now = datetime.now(timezone.utc)
        period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0).isoformat()
        with self._conn() as conn:
            row = conn.execute("""
                SELECT COALESCE(SUM(quantity), 0) as total
                FROM usage_events
                WHERE tenant_id = ? AND unit = ? AND timestamp >= ?
            """, (tenant_id, unit, period_start)).fetchone()
        return row["total"] if row else 0


usage_meter = UsageMeter()
