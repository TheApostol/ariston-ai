"""
Self-Improvement Agent — Ariston AI.

Reads the GxP audit trail (last 100 jobs), identifies patterns (slow jobs,
failed safety checks, frequent topics), generates improvement suggestions via
Claude Haiku, and stores them in the `improvement_log` SQLite table.

Endpoints registered in app/agents/router.py:
  POST /agents/self-improve         — trigger one improvement cycle
  GET  /agents/self-improve/report  — last 10 suggestions
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

logger = logging.getLogger("ariston.self_improvement")

# Shared database paths
_ARISTON_DB = os.path.join("data", "ariston.db")
_GXP_DB     = os.path.join("data", "gxp_audit.db")
_IMP_DB     = os.path.join("data", "ariston.db")   # same DB, different table


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def _ensure_improvement_table() -> None:
    """Create the improvement_log table if it does not exist."""
    with sqlite3.connect(_IMP_DB) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS improvement_log (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp  TEXT NOT NULL,
                category   TEXT NOT NULL,
                suggestion TEXT NOT NULL,
                applied    INTEGER NOT NULL DEFAULT 0
            )
        """)
        conn.commit()


def _read_recent_audit_jobs(limit: int = 100) -> List[Dict[str, Any]]:
    """Pull last `limit` rows from the GxP audit trail."""
    try:
        with sqlite3.connect(_GXP_DB) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM audit_log ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]
    except Exception as e:
        logger.warning("[SelfImprovement] Could not read audit DB: %s", e)
        return []


def _store_suggestion(category: str, suggestion: str) -> int:
    """Insert one improvement suggestion; returns new row id."""
    _ensure_improvement_table()
    ts = datetime.utcnow().isoformat()
    with sqlite3.connect(_IMP_DB) as conn:
        cur = conn.execute(
            "INSERT INTO improvement_log (timestamp, category, suggestion, applied) VALUES (?,?,?,0)",
            (ts, category, suggestion),
        )
        conn.commit()
        return cur.lastrowid


def _fetch_suggestions(limit: int = 10) -> List[Dict[str, Any]]:
    """Return last `limit` suggestions from improvement_log."""
    _ensure_improvement_table()
    with sqlite3.connect(_IMP_DB) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM improvement_log ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


def _mark_applied(suggestion_id: int) -> None:
    """Mark a suggestion as applied."""
    with sqlite3.connect(_IMP_DB) as conn:
        conn.execute("UPDATE improvement_log SET applied=1 WHERE id=?", (suggestion_id,))
        conn.commit()


# ---------------------------------------------------------------------------
# Pattern analysis helpers
# ---------------------------------------------------------------------------

def _analyze_patterns(jobs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Identify slow jobs, safety failures, and frequent topics."""
    slow_threshold_ms = 3000
    slow_jobs  = []
    failed_safety = []
    topic_words: Dict[str, int] = {}

    for job in jobs:
        # Latency check
        latency = job.get("latency_ms") or 0
        if latency > slow_threshold_ms:
            slow_jobs.append({"job_id": job.get("job_id"), "latency_ms": latency})

        # Safety check
        safety_raw = job.get("safety_flag") or ""
        if safety_raw and safety_raw.upper() not in ("SAFE", ""):
            failed_safety.append({"job_id": job.get("job_id"), "safety_flag": safety_raw})

        # Topic frequency — tokenise the prompt field (if present)
        prompt_text = job.get("prompt") or ""
        for word in prompt_text.lower().split():
            if len(word) > 4:
                topic_words[word] = topic_words.get(word, 0) + 1

    top_topics = sorted(topic_words.items(), key=lambda x: x[1], reverse=True)[:10]

    return {
        "total_jobs": len(jobs),
        "slow_jobs_count": len(slow_jobs),
        "slow_jobs_sample": slow_jobs[:5],
        "failed_safety_count": len(failed_safety),
        "failed_safety_sample": failed_safety[:5],
        "top_topics": top_topics,
        "avg_latency_ms": (
            sum(j.get("latency_ms") or 0 for j in jobs) / len(jobs) if jobs else 0
        ),
    }


# ---------------------------------------------------------------------------
# Claude Haiku suggestion generation
# ---------------------------------------------------------------------------

async def _generate_suggestions_with_claude(patterns: Dict[str, Any]) -> List[Dict[str, str]]:
    """Call Claude Haiku to generate improvement suggestions from patterns."""
    try:
        import anthropic as _anthropic
        client = _anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

        summary = json.dumps(patterns, indent=2)
        prompt = (
            f"You are an AI platform optimization expert. Analyse the following usage patterns "
            f"from a medical AI platform's last 100 jobs and generate 3-5 concrete improvement "
            f"suggestions. For each suggestion provide: category (one of: latency, safety, "
            f"prompt_quality, rag_coverage, model_selection) and a 1-2 sentence suggestion.\n\n"
            f"Patterns:\n{summary}\n\n"
            f"Respond ONLY with a JSON array: "
            f'[{{"category":"...", "suggestion":"..."}}]'
        )

        message = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text.strip()
        # Extract JSON array from response
        start = raw.find("[")
        end   = raw.rfind("]") + 1
        if start >= 0 and end > start:
            return json.loads(raw[start:end])
        return []
    except Exception as e:
        logger.warning("[SelfImprovement] Claude call failed: %s", e)
        # Fallback: rule-based suggestions
        suggestions = []
        if patterns.get("slow_jobs_count", 0) > 5:
            suggestions.append({
                "category": "latency",
                "suggestion": (
                    f"{patterns['slow_jobs_count']} jobs exceeded 3 s. "
                    "Consider enabling response caching for repeated queries and "
                    "reducing RAG chunk count for non-clinical layers."
                ),
            })
        if patterns.get("failed_safety_count", 0) > 0:
            suggestions.append({
                "category": "safety",
                "suggestion": (
                    f"{patterns['failed_safety_count']} jobs triggered safety flags. "
                    "Review prompt templates for the affected layers and tighten "
                    "the safety guardrail thresholds."
                ),
            })
        if patterns.get("avg_latency_ms", 0) > 2000:
            suggestions.append({
                "category": "model_selection",
                "suggestion": (
                    "Average latency is above 2 s. Consider routing general-purpose "
                    "queries to claude-haiku-4-5 instead of Sonnet for a 3-5× speed gain."
                ),
            })
        if not suggestions:
            suggestions.append({
                "category": "rag_coverage",
                "suggestion": (
                    "Platform is operating normally. Consider expanding PubMed fetch "
                    "depth from 5 to 8 results for pharma and clinical layers to "
                    "improve RAG grounding scores."
                ),
            })
        return suggestions


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class SelfImprovementAgent:
    """Run improvement cycles and expose reports."""

    def __init__(self) -> None:
        _ensure_improvement_table()

    async def run_cycle(self) -> Dict[str, Any]:
        """
        Execute one improvement cycle:
          1. Read last 100 audit jobs
          2. Analyse patterns
          3. Generate suggestions via Claude Haiku
          4. Store in improvement_log
        Returns a summary dict.
        """
        t0 = time.time()
        jobs     = _read_recent_audit_jobs(100)
        patterns = _analyze_patterns(jobs)
        suggestions = await _generate_suggestions_with_claude(patterns)

        stored_ids = []
        for s in suggestions:
            row_id = _store_suggestion(
                category   = s.get("category", "general"),
                suggestion = s.get("suggestion", ""),
            )
            stored_ids.append(row_id)

        elapsed = round((time.time() - t0) * 1000)
        logger.info(
            "[SelfImprovement] cycle complete: jobs=%d suggestions=%d elapsed_ms=%d",
            len(jobs), len(stored_ids), elapsed,
        )
        return {
            "status": "ok",
            "jobs_analysed": len(jobs),
            "patterns": patterns,
            "suggestions_generated": len(stored_ids),
            "suggestion_ids": stored_ids,
            "elapsed_ms": elapsed,
        }

    def get_report(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Return last `limit` improvement suggestions."""
        return _fetch_suggestions(limit)

    def apply_suggestion(self, suggestion_id: int, prompt_template_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Mark a suggestion as applied and optionally log the target template key.
        Full template patching would be wired here in a production deployment.
        """
        _mark_applied(suggestion_id)
        logger.info("[SelfImprovement] suggestion %d marked as applied (template=%s)",
                    suggestion_id, prompt_template_key)
        return {
            "status": "applied",
            "suggestion_id": suggestion_id,
            "prompt_template_key": prompt_template_key,
        }


# Singleton
self_improvement_agent = SelfImprovementAgent()
