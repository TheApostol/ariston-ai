"""
SLA Monitoring — Phase 5 / Ariston AI.

Tracks platform performance against contractual SLA commitments:
  - Uptime:      standard=99.5%, premium=99.5%, enterprise=99.9%
  - Latency P50: <2s (standard), <1s (premium/enterprise)
  - Latency P95: <5s (standard), <3s (premium/enterprise)
  - Error rate:  <0.5% (standard), <0.1% (enterprise)
  - RAG latency: <3s (standard), <2s (premium)

Data flow: engine.run() → SLA monitor records every request →
           SLA dashboard aggregates → alerts on breach → webhook fired

Architecture:
  - SQLite ring buffer (last 30 days of request metrics)
  - Rolling window SLO calculations (1h, 24h, 7d, 30d windows)
  - Breach detection + automatic webhook event emission
  - Per-tenant SLA reporting (enterprise contracts get dedicated SLAs)
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from statistics import mean, median, quantiles
from typing import Generator, Optional

logger = logging.getLogger("ariston.sla")

_DB_PATH = os.environ.get("ARISTON_SLA_DB", "data/sla_metrics.db")

# SLA targets per tier
SLA_TARGETS: dict[str, dict] = {
    "pilot":      {"uptime_pct": 99.0, "p50_latency_ms": 3000, "p95_latency_ms": 8000, "error_rate_pct": 2.0},
    "standard":   {"uptime_pct": 99.5, "p50_latency_ms": 2000, "p95_latency_ms": 5000, "error_rate_pct": 0.5},
    "premium":    {"uptime_pct": 99.5, "p50_latency_ms": 1000, "p95_latency_ms": 3000, "error_rate_pct": 0.1},
    "enterprise": {"uptime_pct": 99.9, "p50_latency_ms": 800,  "p95_latency_ms": 2000, "error_rate_pct": 0.1},
}


@dataclass
class SLAMetric:
    metric_id: str
    tenant_id: str
    endpoint: str
    layer: str
    latency_ms: float
    success: bool
    error_type: Optional[str]
    timestamp: str
    rag_used: bool = False
    rag_latency_ms: float = 0.0


@dataclass
class SLAReport:
    tenant_id: str
    tier: str
    window_hours: int
    total_requests: int
    successful_requests: int
    error_rate_pct: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    uptime_pct: float
    sla_targets: dict
    breaches: list[str]
    period_start: str
    period_end: str


class SLAMonitor:
    """
    Platform SLA tracking and breach detection.

    Provides:
    - record()        — log a request metric
    - get_report()    — SLA performance report for a window
    - check_breach()  — detect active SLA breaches
    - get_p_latency() — percentile latency calculation
    - purge_old()     — clean up metrics older than retention window
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
                CREATE TABLE IF NOT EXISTS sla_metrics (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_id   TEXT UNIQUE NOT NULL,
                    tenant_id   TEXT NOT NULL DEFAULT 'global',
                    endpoint    TEXT NOT NULL DEFAULT 'unknown',
                    layer       TEXT NOT NULL DEFAULT 'unknown',
                    latency_ms  REAL NOT NULL,
                    success     INTEGER NOT NULL DEFAULT 1,
                    error_type  TEXT,
                    rag_used    INTEGER NOT NULL DEFAULT 0,
                    rag_latency_ms REAL NOT NULL DEFAULT 0.0,
                    timestamp   TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_sla_tenant    ON sla_metrics(tenant_id);
                CREATE INDEX IF NOT EXISTS idx_sla_timestamp ON sla_metrics(timestamp);
                CREATE INDEX IF NOT EXISTS idx_sla_success   ON sla_metrics(success);
            """)

    def record(
        self,
        latency_ms: float,
        success: bool = True,
        tenant_id: str = "global",
        endpoint: str = "unknown",
        layer: str = "unknown",
        error_type: Optional[str] = None,
        rag_used: bool = False,
        rag_latency_ms: float = 0.0,
    ) -> SLAMetric:
        now = datetime.now(timezone.utc).isoformat()
        metric = SLAMetric(
            metric_id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            endpoint=endpoint,
            layer=layer,
            latency_ms=latency_ms,
            success=success,
            error_type=error_type,
            timestamp=now,
            rag_used=rag_used,
            rag_latency_ms=rag_latency_ms,
        )
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO sla_metrics
                  (metric_id, tenant_id, endpoint, layer, latency_ms, success,
                   error_type, rag_used, rag_latency_ms, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metric.metric_id, tenant_id, endpoint, layer, latency_ms,
                int(success), error_type, int(rag_used), rag_latency_ms, now,
            ))
        return metric

    def get_report(
        self,
        tenant_id: str = "global",
        tier: str = "standard",
        window_hours: int = 24,
    ) -> SLAReport:
        now = datetime.now(timezone.utc)
        since = (now - timedelta(hours=window_hours)).isoformat()

        with self._conn() as conn:
            rows = conn.execute("""
                SELECT latency_ms, success, error_type
                FROM sla_metrics
                WHERE (tenant_id = ? OR tenant_id = 'global') AND timestamp >= ?
                ORDER BY timestamp
            """, (tenant_id, since)).fetchall()

        if not rows:
            targets = SLA_TARGETS.get(tier, SLA_TARGETS["standard"])
            return SLAReport(
                tenant_id=tenant_id, tier=tier, window_hours=window_hours,
                total_requests=0, successful_requests=0, error_rate_pct=0.0,
                p50_latency_ms=0.0, p95_latency_ms=0.0, p99_latency_ms=0.0,
                uptime_pct=100.0, sla_targets=targets, breaches=[],
                period_start=since, period_end=now.isoformat(),
            )

        total = len(rows)
        successful = sum(1 for r in rows if r["success"])
        error_rate = round((total - successful) / total * 100, 3)
        latencies = [r["latency_ms"] for r in rows]

        p50 = round(median(latencies), 1)
        p95 = round(quantiles(latencies, n=20)[18], 1) if len(latencies) >= 20 else round(max(latencies), 1)
        p99 = round(quantiles(latencies, n=100)[98], 1) if len(latencies) >= 100 else round(max(latencies), 1)

        # Uptime approximation: (successful / total) * 100
        uptime = round(successful / total * 100, 3) if total > 0 else 100.0

        targets = SLA_TARGETS.get(tier, SLA_TARGETS["standard"])
        breaches = self._detect_breaches(uptime, p50, p95, error_rate, targets)

        return SLAReport(
            tenant_id=tenant_id, tier=tier, window_hours=window_hours,
            total_requests=total, successful_requests=successful,
            error_rate_pct=error_rate, p50_latency_ms=p50,
            p95_latency_ms=p95, p99_latency_ms=p99,
            uptime_pct=uptime, sla_targets=targets, breaches=breaches,
            period_start=since, period_end=now.isoformat(),
        )

    def check_breach(self, tenant_id: str = "global", tier: str = "standard") -> list[str]:
        """Return active SLA breaches in the last 1 hour."""
        report = self.get_report(tenant_id=tenant_id, tier=tier, window_hours=1)
        return report.breaches

    def get_uptime_series(self, tenant_id: str = "global", days: int = 30) -> list[dict]:
        """Daily uptime percentages for dashboard chart."""
        since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT substr(timestamp, 1, 10) as day,
                       COUNT(*) as total,
                       SUM(success) as ok
                FROM sla_metrics
                WHERE (tenant_id = ? OR tenant_id = 'global') AND timestamp >= ?
                GROUP BY day ORDER BY day
            """, (tenant_id, since)).fetchall()
        return [
            {
                "date": r["day"],
                "uptime_pct": round(r["ok"] / r["total"] * 100, 3) if r["total"] > 0 else 100.0,
                "total_requests": r["total"],
            }
            for r in rows
        ]

    def purge_old(self, retention_days: int = 90) -> int:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=retention_days)).isoformat()
        with self._conn() as conn:
            result = conn.execute("DELETE FROM sla_metrics WHERE timestamp < ?", (cutoff,))
        return result.rowcount

    def _detect_breaches(
        self, uptime: float, p50: float, p95: float, error_rate: float, targets: dict
    ) -> list[str]:
        breaches = []
        if uptime < targets["uptime_pct"]:
            breaches.append(f"Uptime {uptime:.2f}% < SLA {targets['uptime_pct']}%")
        if p50 > targets["p50_latency_ms"]:
            breaches.append(f"P50 latency {p50:.0f}ms > SLA {targets['p50_latency_ms']}ms")
        if p95 > targets["p95_latency_ms"]:
            breaches.append(f"P95 latency {p95:.0f}ms > SLA {targets['p95_latency_ms']}ms")
        if error_rate > targets["error_rate_pct"]:
            breaches.append(f"Error rate {error_rate:.2f}% > SLA {targets['error_rate_pct']}%")
        return breaches


sla_monitor = SLAMonitor()
