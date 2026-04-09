"""
Feedback Loop — customer ratings → improvement signals.

Persists feedback to SQLite and aggregates signals for the
benchmark analyzer and improvement agent.
"""

import json
import sqlite3
from datetime import datetime, timezone
from typing import List, Optional

DB_PATH = "benchmarks/feedback.db"


def _get_conn(db_path: str = DB_PATH) -> sqlite3.Connection:
    import os
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db(db_path: str = DB_PATH):
    with _get_conn(db_path) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT,
                layer TEXT,
                model TEXT,
                rating INTEGER NOT NULL,          -- 1-5
                nps_score INTEGER,                -- 0-10
                feature_ratings TEXT,             -- JSON: {feature: score}
                comment TEXT,
                blockers TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS improvement_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_type TEXT NOT NULL,        -- low_rating | safety_failure | latency
                target TEXT NOT NULL,             -- layer or model name
                details TEXT,                     -- JSON
                processed INTEGER DEFAULT 0,
                created_at TEXT NOT NULL
            );
        """)
        conn.commit()


def submit_feedback(
    rating: int,
    job_id: Optional[str] = None,
    layer: Optional[str] = None,
    model: Optional[str] = None,
    nps_score: Optional[int] = None,
    feature_ratings: Optional[dict] = None,
    comment: Optional[str] = None,
    blockers: Optional[str] = None,
    db_path: str = DB_PATH,
) -> dict:
    """
    Submit user feedback for a completed AI job.

    Args:
        rating: Overall satisfaction 1-5
        job_id: The engine job_id from response metadata
        layer: The AI layer used (clinical, pharma, etc.)
        model: The model that handled the request
        nps_score: Net Promoter Score 0-10
        feature_ratings: dict of {feature_name: 1-5}
        comment: Free-text comment
        blockers: Free-text description of blockers/issues
    """
    _init_db(db_path)

    created_at = datetime.now(timezone.utc).isoformat()

    with _get_conn(db_path) as conn:
        cursor = conn.execute(
            """INSERT INTO feedback
               (job_id, layer, model, rating, nps_score, feature_ratings, comment, blockers, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                job_id, layer, model, rating, nps_score,
                json.dumps(feature_ratings) if feature_ratings else None,
                comment, blockers, created_at,
            ),
        )
        feedback_id = cursor.lastrowid

        # Auto-generate improvement signal for low ratings
        if rating <= 2:
            target = layer or model or "unknown"
            conn.execute(
                """INSERT INTO improvement_signals
                   (signal_type, target, details, created_at)
                   VALUES (?, ?, ?, ?)""",
                (
                    "low_rating",
                    target,
                    json.dumps({
                        "feedback_id": feedback_id,
                        "rating": rating,
                        "comment": comment,
                        "blockers": blockers,
                    }),
                    created_at,
                ),
            )
        conn.commit()

    return {
        "feedback_id": feedback_id,
        "status": "recorded",
        "signal_generated": rating <= 2,
        "created_at": created_at,
    }


def get_feedback_summary(db_path: str = DB_PATH) -> dict:
    """Aggregate feedback statistics for the metrics dashboard."""
    _init_db(db_path)

    with _get_conn(db_path) as conn:
        rows = conn.execute("SELECT * FROM feedback ORDER BY created_at DESC").fetchall()

    if not rows:
        return {
            "total_responses": 0,
            "avg_rating": None,
            "avg_nps": None,
            "rating_distribution": {},
            "by_layer": {},
            "recent_comments": [],
        }

    total = len(rows)
    ratings = [r["rating"] for r in rows if r["rating"]]
    nps_scores = [r["nps_score"] for r in rows if r["nps_score"] is not None]

    rating_dist = {}
    for r in range(1, 6):
        rating_dist[str(r)] = sum(1 for rt in ratings if rt == r)

    # Aggregate by layer
    by_layer: dict = {}
    for row in rows:
        layer = row["layer"] or "unknown"
        if layer not in by_layer:
            by_layer[layer] = {"ratings": [], "count": 0}
        if row["rating"]:
            by_layer[layer]["ratings"].append(row["rating"])
            by_layer[layer]["count"] += 1

    by_layer_summary = {
        layer: {
            "avg_rating": round(sum(d["ratings"]) / len(d["ratings"]), 2) if d["ratings"] else None,
            "count": d["count"],
        }
        for layer, d in by_layer.items()
    }

    recent_comments = [
        {
            "comment": r["comment"],
            "rating": r["rating"],
            "layer": r["layer"],
            "created_at": r["created_at"],
        }
        for r in rows[:5]
        if r["comment"]
    ]

    return {
        "total_responses": total,
        "avg_rating": round(sum(ratings) / len(ratings), 2) if ratings else None,
        "avg_nps": round(sum(nps_scores) / len(nps_scores), 1) if nps_scores else None,
        "rating_distribution": rating_dist,
        "by_layer": by_layer_summary,
        "recent_comments": recent_comments,
    }


def get_unprocessed_signals(db_path: str = DB_PATH) -> List[dict]:
    """Return unprocessed improvement signals for the improvement agent."""
    _init_db(db_path)

    with _get_conn(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM improvement_signals WHERE processed = 0 ORDER BY created_at ASC"
        ).fetchall()

    return [dict(r) for r in rows]


def mark_signal_processed(signal_id: int, db_path: str = DB_PATH):
    """Mark an improvement signal as processed after the agent has handled it."""
    _init_db(db_path)

    with _get_conn(db_path) as conn:
        conn.execute(
            "UPDATE improvement_signals SET processed = 1 WHERE id = ?",
            (signal_id,),
        )
        conn.commit()
