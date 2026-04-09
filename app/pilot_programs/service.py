"""
Pilot Program — Database service.

SQLite-backed storage for pilots, document versions, ROI metrics, and feedback.
Supports multi-tenancy: each pilot is isolated by pilot_id.
"""

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import List, Optional

DB_PATH = "data/pilots.db"


def _get_conn(db_path: str = DB_PATH) -> sqlite3.Connection:
    import os
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db(db_path: str = DB_PATH):
    with _get_conn(db_path) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS pilots (
                id TEXT PRIMARY KEY,
                company_name TEXT NOT NULL,
                contact_name TEXT NOT NULL,
                contact_email TEXT NOT NULL,
                country TEXT NOT NULL,
                locale TEXT NOT NULL,
                agency TEXT NOT NULL,
                therapeutic_area TEXT NOT NULL,
                commitment_level TEXT NOT NULL,     -- trial | committed | enterprise
                status TEXT NOT NULL DEFAULT 'active',
                enrolled_at TEXT NOT NULL,
                notes TEXT,
                metadata TEXT                       -- JSON for extensibility
            );

            CREATE TABLE IF NOT EXISTS document_versions (
                id TEXT PRIMARY KEY,
                pilot_id TEXT NOT NULL,
                document_type TEXT NOT NULL,
                version INTEGER NOT NULL DEFAULT 1,
                drug_name TEXT,
                indication TEXT,
                content TEXT NOT NULL,
                language TEXT DEFAULT 'en',
                agency TEXT,
                created_at TEXT NOT NULL,
                created_by TEXT,
                change_summary TEXT,
                is_active INTEGER DEFAULT 1,
                FOREIGN KEY (pilot_id) REFERENCES pilots(id)
            );

            CREATE TABLE IF NOT EXISTS roi_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pilot_id TEXT NOT NULL,
                metric_date TEXT NOT NULL,
                document_type TEXT,
                manual_hours_baseline REAL,         -- hours without AI
                ai_assisted_hours REAL,             -- hours with AI
                time_saved_hours REAL,
                hourly_rate_usd REAL DEFAULT 150,
                cost_saved_usd REAL,
                documents_generated INTEGER DEFAULT 0,
                user_sessions INTEGER DEFAULT 0,
                metadata TEXT,                      -- JSON
                FOREIGN KEY (pilot_id) REFERENCES pilots(id)
            );

            CREATE TABLE IF NOT EXISTS pilot_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pilot_id TEXT NOT NULL,
                rating INTEGER,                     -- 1-5
                nps_score INTEGER,                  -- 0-10
                feature_ratings TEXT,               -- JSON
                comment TEXT,
                blockers TEXT,
                feature_requests TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (pilot_id) REFERENCES pilots(id)
            );
        """)
        conn.commit()


# ── Pilot enrollment ──────────────────────────────────────────────────────────

def enroll_pilot(
    company_name: str,
    contact_name: str,
    contact_email: str,
    country: str,
    locale: str,
    agency: str,
    therapeutic_area: str,
    commitment_level: str,
    notes: Optional[str] = None,
    metadata: Optional[dict] = None,
    db_path: str = DB_PATH,
) -> dict:
    """Enroll a new pilot organization."""
    _init_db(db_path)

    pilot_id = str(uuid.uuid4())
    enrolled_at = datetime.now(timezone.utc).isoformat()

    with _get_conn(db_path) as conn:
        conn.execute(
            """INSERT INTO pilots
               (id, company_name, contact_name, contact_email, country, locale,
                agency, therapeutic_area, commitment_level, enrolled_at, notes, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                pilot_id, company_name, contact_name, contact_email,
                country, locale, agency, therapeutic_area, commitment_level,
                enrolled_at, notes, json.dumps(metadata) if metadata else None,
            ),
        )
        conn.commit()

    return {
        "pilot_id": pilot_id,
        "status": "enrolled",
        "company_name": company_name,
        "enrolled_at": enrolled_at,
    }


def get_pilot(pilot_id: str, db_path: str = DB_PATH) -> Optional[dict]:
    """Get a pilot by ID."""
    _init_db(db_path)

    with _get_conn(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM pilots WHERE id = ?", (pilot_id,)
        ).fetchone()

    if not row:
        return None
    result = dict(row)
    if result.get("metadata"):
        result["metadata"] = json.loads(result["metadata"])
    return result


def list_pilots(status: Optional[str] = None, db_path: str = DB_PATH) -> List[dict]:
    """List all pilots, optionally filtered by status."""
    _init_db(db_path)

    with _get_conn(db_path) as conn:
        if status:
            rows = conn.execute(
                "SELECT * FROM pilots WHERE status = ? ORDER BY enrolled_at DESC", (status,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM pilots ORDER BY enrolled_at DESC"
            ).fetchall()

    results = []
    for row in rows:
        r = dict(row)
        if r.get("metadata"):
            r["metadata"] = json.loads(r["metadata"])
        results.append(r)
    return results


def update_pilot_status(pilot_id: str, status: str, db_path: str = DB_PATH) -> bool:
    """Update the status of a pilot (active, paused, completed)."""
    _init_db(db_path)

    with _get_conn(db_path) as conn:
        conn.execute(
            "UPDATE pilots SET status = ? WHERE id = ?", (status, pilot_id)
        )
        conn.commit()
    return True


# ── Document versioning ───────────────────────────────────────────────────────

def save_document_version(
    pilot_id: str,
    document_type: str,
    content: str,
    drug_name: Optional[str] = None,
    indication: Optional[str] = None,
    language: str = "en",
    agency: Optional[str] = None,
    created_by: Optional[str] = None,
    change_summary: Optional[str] = None,
    db_path: str = DB_PATH,
) -> dict:
    """
    Save a new document version for a pilot.

    Each save increments the version number automatically.
    Previous versions are retained for audit/rollback.
    """
    _init_db(db_path)

    doc_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()

    with _get_conn(db_path) as conn:
        # Get next version number
        result = conn.execute(
            """SELECT COALESCE(MAX(version), 0) + 1 as next_version
               FROM document_versions
               WHERE pilot_id = ? AND document_type = ?""",
            (pilot_id, document_type),
        ).fetchone()
        next_version = result["next_version"]

        # Deactivate previous versions
        conn.execute(
            """UPDATE document_versions
               SET is_active = 0
               WHERE pilot_id = ? AND document_type = ?""",
            (pilot_id, document_type),
        )

        # Insert new version
        conn.execute(
            """INSERT INTO document_versions
               (id, pilot_id, document_type, version, drug_name, indication,
                content, language, agency, created_at, created_by, change_summary, is_active)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)""",
            (
                doc_id, pilot_id, document_type, next_version,
                drug_name, indication, content, language, agency,
                created_at, created_by, change_summary,
            ),
        )
        conn.commit()

    return {
        "document_id": doc_id,
        "pilot_id": pilot_id,
        "document_type": document_type,
        "version": next_version,
        "created_at": created_at,
    }


def get_document_versions(
    pilot_id: str,
    document_type: Optional[str] = None,
    active_only: bool = False,
    db_path: str = DB_PATH,
) -> List[dict]:
    """List document versions for a pilot."""
    _init_db(db_path)

    query = "SELECT id, pilot_id, document_type, version, drug_name, indication, language, agency, created_at, created_by, change_summary, is_active FROM document_versions WHERE pilot_id = ?"
    params: list = [pilot_id]

    if document_type:
        query += " AND document_type = ?"
        params.append(document_type)

    if active_only:
        query += " AND is_active = 1"

    query += " ORDER BY version DESC"

    with _get_conn(db_path) as conn:
        rows = conn.execute(query, params).fetchall()

    return [dict(r) for r in rows]


def get_document_content(document_id: str, db_path: str = DB_PATH) -> Optional[dict]:
    """Get the full content of a specific document version."""
    _init_db(db_path)

    with _get_conn(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM document_versions WHERE id = ?", (document_id,)
        ).fetchone()

    return dict(row) if row else None


# ── ROI metrics ───────────────────────────────────────────────────────────────

def record_roi_metric(
    pilot_id: str,
    document_type: str,
    manual_hours_baseline: float,
    ai_assisted_hours: float,
    hourly_rate_usd: float = 150.0,
    documents_generated: int = 1,
    user_sessions: int = 1,
    metadata: Optional[dict] = None,
    db_path: str = DB_PATH,
) -> dict:
    """Record an ROI data point for a pilot."""
    _init_db(db_path)

    time_saved = manual_hours_baseline - ai_assisted_hours
    cost_saved = time_saved * hourly_rate_usd
    metric_date = datetime.now(timezone.utc).date().isoformat()

    with _get_conn(db_path) as conn:
        cursor = conn.execute(
            """INSERT INTO roi_metrics
               (pilot_id, metric_date, document_type, manual_hours_baseline,
                ai_assisted_hours, time_saved_hours, hourly_rate_usd, cost_saved_usd,
                documents_generated, user_sessions, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                pilot_id, metric_date, document_type, manual_hours_baseline,
                ai_assisted_hours, time_saved, hourly_rate_usd, cost_saved,
                documents_generated, user_sessions,
                json.dumps(metadata) if metadata else None,
            ),
        )
        conn.commit()
        metric_id = cursor.lastrowid

    return {
        "metric_id": metric_id,
        "pilot_id": pilot_id,
        "metric_date": metric_date,
        "time_saved_hours": round(time_saved, 2),
        "cost_saved_usd": round(cost_saved, 2),
        "time_reduction_pct": round((time_saved / manual_hours_baseline) * 100, 1) if manual_hours_baseline else 0,
    }


def get_roi_summary(pilot_id: str, db_path: str = DB_PATH) -> dict:
    """Return aggregated ROI summary for a pilot."""
    _init_db(db_path)

    with _get_conn(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM roi_metrics WHERE pilot_id = ? ORDER BY metric_date DESC",
            (pilot_id,),
        ).fetchall()

    if not rows:
        return {
            "pilot_id": pilot_id,
            "total_documents": 0,
            "total_time_saved_hours": 0,
            "total_cost_saved_usd": 0,
            "avg_time_reduction_pct": 0,
            "data_points": 0,
        }

    total_docs = sum(r["documents_generated"] for r in rows)
    total_time_saved = sum(r["time_saved_hours"] for r in rows if r["time_saved_hours"])
    total_cost_saved = sum(r["cost_saved_usd"] for r in rows if r["cost_saved_usd"])
    baseline_hours = sum(r["manual_hours_baseline"] for r in rows if r["manual_hours_baseline"])
    avg_reduction = (total_time_saved / baseline_hours * 100) if baseline_hours else 0

    return {
        "pilot_id": pilot_id,
        "total_documents": total_docs,
        "total_time_saved_hours": round(total_time_saved, 2),
        "total_cost_saved_usd": round(total_cost_saved, 2),
        "avg_time_reduction_pct": round(avg_reduction, 1),
        "data_points": len(rows),
        "latest_date": rows[0]["metric_date"] if rows else None,
    }


# ── Pilot feedback ────────────────────────────────────────────────────────────

def submit_pilot_feedback(
    pilot_id: str,
    rating: Optional[int] = None,
    nps_score: Optional[int] = None,
    feature_ratings: Optional[dict] = None,
    comment: Optional[str] = None,
    blockers: Optional[str] = None,
    feature_requests: Optional[str] = None,
    db_path: str = DB_PATH,
) -> dict:
    """Submit feedback for a pilot program."""
    _init_db(db_path)

    created_at = datetime.now(timezone.utc).isoformat()

    with _get_conn(db_path) as conn:
        cursor = conn.execute(
            """INSERT INTO pilot_feedback
               (pilot_id, rating, nps_score, feature_ratings, comment,
                blockers, feature_requests, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                pilot_id, rating, nps_score,
                json.dumps(feature_ratings) if feature_ratings else None,
                comment, blockers, feature_requests, created_at,
            ),
        )
        conn.commit()
        feedback_id = cursor.lastrowid

    return {
        "feedback_id": feedback_id,
        "pilot_id": pilot_id,
        "created_at": created_at,
        "status": "recorded",
    }


def get_pilot_feedback(pilot_id: str, db_path: str = DB_PATH) -> List[dict]:
    """Get all feedback for a pilot."""
    _init_db(db_path)

    with _get_conn(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM pilot_feedback WHERE pilot_id = ? ORDER BY created_at DESC",
            (pilot_id,),
        ).fetchall()

    results = []
    for row in rows:
        r = dict(row)
        if r.get("feature_ratings"):
            r["feature_ratings"] = json.loads(r["feature_ratings"])
        results.append(r)
    return results


def get_all_pilots_analytics(db_path: str = DB_PATH) -> dict:
    """Return platform-wide analytics across all pilots."""
    _init_db(db_path)

    pilots = list_pilots(db_path=db_path)

    with _get_conn(db_path) as conn:
        total_docs = conn.execute(
            "SELECT COALESCE(SUM(documents_generated), 0) as total FROM roi_metrics"
        ).fetchone()["total"]

        total_saved = conn.execute(
            "SELECT COALESCE(SUM(cost_saved_usd), 0) as total FROM roi_metrics"
        ).fetchone()["total"]

        total_time_saved = conn.execute(
            "SELECT COALESCE(SUM(time_saved_hours), 0) as total FROM roi_metrics"
        ).fetchone()["total"]

        avg_rating_row = conn.execute(
            "SELECT AVG(rating) as avg FROM pilot_feedback WHERE rating IS NOT NULL"
        ).fetchone()

    return {
        "total_pilots": len(pilots),
        "active_pilots": sum(1 for p in pilots if p["status"] == "active"),
        "total_documents_generated": total_docs,
        "total_cost_saved_usd": round(total_saved, 2),
        "total_time_saved_hours": round(total_time_saved, 2),
        "avg_pilot_rating": round(avg_rating_row["avg"], 2) if avg_rating_row["avg"] else None,
        "countries": list({p["country"] for p in pilots}),
        "therapeutic_areas": list({p["therapeutic_area"] for p in pilots}),
    }
