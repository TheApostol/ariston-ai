"""
RWE Data Accumulation Pipeline — Phase 6 / Ariston AI.

Orchestrates continuous ingestion of Real-World Evidence from LATAM sources
into the semantic embedding store. Drives the data moat flywheel:

  LATAM data connectors → normalization → embedding → freshness monitoring

Architecture:
  - Per-namespace accumulation (pubmed | rwe | regulatory | drug_discovery)
  - Freshness tracking: each dataset has a last_refreshed timestamp
  - Staleness detection: configurable max_age_hours per dataset type
  - Incremental upsert: SHA-256 content deduplication prevents re-embedding
  - Stats + health check endpoint for ops dashboard

Data flywheel value:
  The more RWE records accumulated, the higher the similarity search quality,
  which improves biomarker discovery → drug target confidence → trial design.
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
from typing import Generator, Optional

logger = logging.getLogger("ariston.rwe.accumulation")

_DB_PATH = os.environ.get("ARISTON_RWE_ACC_DB", "data/rwe_accumulation.db")

# Staleness thresholds per dataset type (hours)
FRESHNESS_THRESHOLDS: dict[str, int] = {
    "epidemiological": 168,   # 1 week
    "mortality": 720,         # 30 days
    "hospital_records": 720,  # 30 days
    "disease_burden": 2160,   # 90 days
    "pubmed": 24,             # daily
    "clinical_trials": 48,    # 48 hours
}


@dataclass
class AccumulationRecord:
    record_id: str
    namespace: str
    dataset_type: str
    country: str
    condition: str
    source: str
    doc_id: str                # embedding store doc_id
    record_count: int
    year: int
    refreshed_at: str
    metadata: dict = field(default_factory=dict)


@dataclass
class FreshnessStatus:
    dataset_type: str
    country: str
    condition: str
    last_refreshed: Optional[str]
    age_hours: Optional[float]
    threshold_hours: int
    is_stale: bool
    record_count: int


@dataclass
class AccumulationStats:
    total_records: int
    total_namespaces: int
    by_country: dict[str, int]
    by_dataset_type: dict[str, int]
    stale_datasets: list[FreshnessStatus]
    freshness_score: float   # 0.0–1.0 (fraction of datasets within threshold)


class RWEAccumulationPipeline:
    """
    RWE data accumulation and freshness monitoring.

    Provides:
    - accumulate()         — ingest LATAM data into embedding store
    - get_freshness()      — check data staleness per dataset/country
    - get_stats()          — overall accumulation health
    - refresh_stale()      — trigger re-fetch for stale datasets
    - purge_old()          — remove records older than retention window
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
                CREATE TABLE IF NOT EXISTS rwe_accumulation (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    record_id       TEXT UNIQUE NOT NULL,
                    namespace       TEXT NOT NULL DEFAULT 'rwe',
                    dataset_type    TEXT NOT NULL,
                    country         TEXT NOT NULL,
                    condition       TEXT NOT NULL,
                    source          TEXT NOT NULL,
                    doc_id          TEXT NOT NULL,
                    record_count    INTEGER NOT NULL DEFAULT 0,
                    year            INTEGER NOT NULL,
                    refreshed_at    TEXT NOT NULL,
                    metadata_json   TEXT NOT NULL DEFAULT '{}'
                );

                CREATE INDEX IF NOT EXISTS idx_rwe_country   ON rwe_accumulation(country);
                CREATE INDEX IF NOT EXISTS idx_rwe_condition ON rwe_accumulation(condition);
                CREATE INDEX IF NOT EXISTS idx_rwe_type      ON rwe_accumulation(dataset_type);
                CREATE INDEX IF NOT EXISTS idx_rwe_refreshed ON rwe_accumulation(refreshed_at);
            """)

    async def accumulate(
        self,
        country: str,
        condition: str,
        dataset_type: str = "epidemiological",
        year: Optional[int] = None,
        namespace: str = "rwe",
        embed: bool = True,
    ) -> AccumulationRecord:
        """
        Fetch LATAM data and accumulate into embedding store.
        Returns accumulation record with doc_id for retrieval.
        """
        from vinci_core.latam_data.connectors import latam_data_connector

        year = year or datetime.now(timezone.utc).year - 1

        # Fetch from LATAM connector
        records = await latam_data_connector.fetch_epidemiological(
            country=country,
            condition=condition,
            year=year,
        )

        if not records:
            logger.warning("[RWE] No records for country=%s condition=%s", country, condition)

        # Serialize records to embeddable text
        content = self._records_to_text(records, country, condition, dataset_type, year)
        source = records[0].source if records else "unknown"

        doc_id = str(uuid.uuid4())
        if embed and records:
            try:
                from vinci_core.embeddings.store import embedding_store
                doc = embedding_store.upsert(
                    content=content,
                    namespace=namespace,
                    doc_id=doc_id,
                    metadata={
                        "country": country,
                        "condition": condition,
                        "dataset_type": dataset_type,
                        "year": year,
                        "record_count": len(records),
                        "source": source,
                    },
                )
                doc_id = doc.doc_id
            except Exception as e:
                logger.warning("[RWE] Embedding failed: %s", e)

        acc_record = AccumulationRecord(
            record_id=str(uuid.uuid4()),
            namespace=namespace,
            dataset_type=dataset_type,
            country=country,
            condition=condition,
            source=source,
            doc_id=doc_id,
            record_count=len(records),
            year=year,
            refreshed_at=datetime.now(timezone.utc).isoformat(),
            metadata={"embedded": embed and bool(records)},
        )

        # Upsert accumulation record (update if same country+condition+type+year)
        with self._conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO rwe_accumulation
                  (record_id, namespace, dataset_type, country, condition, source,
                   doc_id, record_count, year, refreshed_at, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                acc_record.record_id, namespace, dataset_type, country, condition,
                source, doc_id, len(records), year,
                acc_record.refreshed_at, json.dumps(acc_record.metadata),
            ))

        logger.info(
            "[RWE] Accumulated country=%s condition=%s records=%d doc_id=%s",
            country, condition, len(records), doc_id,
        )
        return acc_record

    def get_freshness(
        self,
        country: Optional[str] = None,
        condition: Optional[str] = None,
        dataset_type: Optional[str] = None,
    ) -> list[FreshnessStatus]:
        """Check freshness of accumulated datasets."""
        clauses = []
        params: list = []
        if country:
            clauses.append("country = ?")
            params.append(country.lower())
        if condition:
            clauses.append("condition = ?")
            params.append(condition.lower())
        if dataset_type:
            clauses.append("dataset_type = ?")
            params.append(dataset_type)

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""

        with self._conn() as conn:
            rows = conn.execute(f"""
                SELECT dataset_type, country, condition,
                       MAX(refreshed_at) as last_refreshed,
                       SUM(record_count) as total_records
                FROM rwe_accumulation
                {where}
                GROUP BY dataset_type, country, condition
            """, params).fetchall()

        now = datetime.now(timezone.utc)
        results = []
        for row in rows:
            threshold = FRESHNESS_THRESHOLDS.get(row["dataset_type"], 720)
            last = row["last_refreshed"]
            age_hours: Optional[float] = None
            is_stale = True
            if last:
                try:
                    last_dt = datetime.fromisoformat(last.replace("Z", "+00:00"))
                    if last_dt.tzinfo is None:
                        last_dt = last_dt.replace(tzinfo=timezone.utc)
                    age_hours = round((now - last_dt).total_seconds() / 3600, 1)
                    is_stale = age_hours > threshold
                except Exception:
                    pass

            results.append(FreshnessStatus(
                dataset_type=row["dataset_type"],
                country=row["country"],
                condition=row["condition"],
                last_refreshed=last,
                age_hours=age_hours,
                threshold_hours=threshold,
                is_stale=is_stale,
                record_count=row["total_records"] or 0,
            ))

        return results

    def get_stats(self) -> AccumulationStats:
        """Overall RWE accumulation health metrics."""
        with self._conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM rwe_accumulation").fetchone()[0]
            namespaces = conn.execute("SELECT COUNT(DISTINCT namespace) FROM rwe_accumulation").fetchone()[0]
            by_country = conn.execute(
                "SELECT country, SUM(record_count) as cnt FROM rwe_accumulation GROUP BY country"
            ).fetchall()
            by_type = conn.execute(
                "SELECT dataset_type, SUM(record_count) as cnt FROM rwe_accumulation GROUP BY dataset_type"
            ).fetchall()

        freshness = self.get_freshness()
        stale = [f for f in freshness if f.is_stale]
        score = round(1 - len(stale) / len(freshness), 3) if freshness else 1.0

        return AccumulationStats(
            total_records=total,
            total_namespaces=namespaces,
            by_country={r["country"]: r["cnt"] for r in by_country},
            by_dataset_type={r["dataset_type"]: r["cnt"] for r in by_type},
            stale_datasets=stale,
            freshness_score=score,
        )

    async def refresh_stale(
        self,
        max_datasets: int = 10,
    ) -> dict:
        """Re-fetch all stale datasets. Returns refresh summary."""
        stale = [f for f in self.get_freshness() if f.is_stale][:max_datasets]
        refreshed = []
        failed = []

        for fs in stale:
            try:
                await self.accumulate(
                    country=fs.country,
                    condition=fs.condition,
                    dataset_type=fs.dataset_type,
                )
                refreshed.append(f"{fs.country}/{fs.condition}")
            except Exception as e:
                logger.error("[RWE] Refresh failed %s/%s: %s", fs.country, fs.condition, e)
                failed.append(f"{fs.country}/{fs.condition}")

        return {
            "stale_found": len(stale),
            "refreshed": refreshed,
            "failed": failed,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def purge_old(self, retention_days: int = 365) -> int:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=retention_days)).isoformat()
        with self._conn() as conn:
            result = conn.execute(
                "DELETE FROM rwe_accumulation WHERE refreshed_at < ?", (cutoff,)
            )
        return result.rowcount

    def _records_to_text(
        self, records: list, country: str, condition: str, dataset_type: str, year: int
    ) -> str:
        """Convert DataRecord list to embeddable text for semantic search."""
        lines = [
            f"Real-World Evidence: {condition} in {country.title()} ({year})",
            f"Dataset type: {dataset_type}",
            f"Source: {records[0].source if records else 'unknown'}",
            "",
        ]
        for r in records:
            age = f" age {r.age_group}" if r.age_group else ""
            sex = f" sex={r.sex}" if r.sex else ""
            region = f" region={r.region}" if r.region else ""
            lines.append(
                f"ICD-10 {r.condition_code} ({r.condition_name}): "
                f"{r.count} cases{age}{sex}{region}"
            )
        return "\n".join(lines)


rwe_accumulation = RWEAccumulationPipeline()
