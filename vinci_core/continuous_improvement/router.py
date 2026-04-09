"""
Continuous Improvement — FastAPI Router.

Endpoints:
  GET  /improvement/benchmarks/analyze    — analyze eval_logs for low-scoring patterns
  GET  /improvement/benchmarks/patterns   — get recent low-scoring log entries
  POST /improvement/cycle/run             — run one full improvement cycle
  GET  /improvement/cycle/history         — improvement cycle logs
  POST /improvement/feedback              — submit user feedback
  GET  /improvement/feedback/summary      — aggregated feedback stats
  GET  /improvement/metrics/dashboard     — combined metrics dashboard
  GET  /improvement/health                — health check
"""

from fastapi import APIRouter, BackgroundTasks, Query
from pydantic import BaseModel
from typing import Optional

from .benchmark_analyzer import analyze_benchmarks, get_low_scoring_patterns
from .improvement_agent import run_improvement_cycle, get_improvement_history
from .feedback_loop import submit_feedback, get_feedback_summary

router = APIRouter(prefix="/improvement", tags=["Autonomous Improvement Loop"])


# ── Request models ────────────────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    rating: int                           # 1-5 overall rating
    job_id: Optional[str] = None
    layer: Optional[str] = None
    model: Optional[str] = None
    nps_score: Optional[int] = None       # 0-10
    feature_ratings: Optional[dict] = None
    comment: Optional[str] = None
    blockers: Optional[str] = None


# ── Benchmark analysis ────────────────────────────────────────────────────────

@router.get("/benchmarks/analyze")
async def analyze(
    threshold: float = Query(0.75, description="Success rate threshold for flagging (default 0.75)"),
):
    """
    Analyze evaluation logs for low-scoring patterns.

    Flags layers and models with combined score < threshold.
    Combined score = safety(50%) + grounding(30%) + confidence(20%).
    """
    return analyze_benchmarks(threshold=threshold)


@router.get("/benchmarks/patterns")
async def low_scoring_patterns(
    threshold: float = Query(0.75, description="Score threshold"),
    limit: int = Query(10, description="Max number of entries to return"),
):
    """Return recent low-scoring benchmark log entries for review."""
    patterns = get_low_scoring_patterns(threshold=threshold, limit=limit)
    return {
        "patterns": patterns,
        "total": len(patterns),
        "threshold": threshold,
    }


# ── Improvement cycle ─────────────────────────────────────────────────────────

@router.post("/cycle/run")
async def trigger_improvement_cycle(background_tasks: BackgroundTasks):
    """
    Trigger one full autonomous improvement cycle.

    The cycle:
      1. Analyzes benchmark logs for low-scoring patterns
      2. Reads unprocessed customer feedback signals
      3. Generates a JSON improvement plan via LLM
      4. Logs the plan to benchmarks/improvement_log.jsonl
      5. Marks feedback signals as processed

    Returns immediately with job status; cycle runs in background.
    """
    async def _run():
        try:
            await run_improvement_cycle()
        except Exception as e:
            print(f"[ImprovementCycle] Failed: {e}")

    background_tasks.add_task(_run)
    return {
        "status": "improvement_cycle_started",
        "message": "Cycle running in background. Check /improvement/cycle/history for results.",
    }


@router.post("/cycle/run/sync")
async def trigger_improvement_cycle_sync():
    """
    Run one full autonomous improvement cycle synchronously (waits for completion).

    Use for testing or when you need the plan immediately.
    """
    return await run_improvement_cycle()


@router.get("/cycle/history")
async def improvement_history(
    limit: int = Query(20, description="Max number of historical cycles to return"),
):
    """Return history of improvement cycles with their generated plans."""
    history = get_improvement_history(limit=limit)
    return {
        "cycles": history,
        "total": len(history),
    }


# ── Feedback ──────────────────────────────────────────────────────────────────

@router.post("/feedback")
async def record_feedback(request: FeedbackRequest):
    """
    Submit user feedback for a completed AI interaction.

    Low ratings (1-2) automatically generate improvement signals
    that feed into the next improvement cycle.

    Example:
      POST /api/v1/improvement/feedback
      { "rating": 2, "job_id": "...", "layer": "pharma", "comment": "Response was too vague" }
    """
    return submit_feedback(
        rating=request.rating,
        job_id=request.job_id,
        layer=request.layer,
        model=request.model,
        nps_score=request.nps_score,
        feature_ratings=request.feature_ratings,
        comment=request.comment,
        blockers=request.blockers,
    )


@router.get("/feedback/summary")
async def feedback_summary():
    """Return aggregated feedback statistics."""
    return get_feedback_summary()


# ── Combined metrics dashboard ────────────────────────────────────────────────

@router.get("/metrics/dashboard")
async def metrics_dashboard():
    """
    Combined metrics dashboard showing:
    - Benchmark performance by layer and model
    - Customer feedback summary
    - Improvement cycle activity
    - Trend indicators
    """
    analysis = analyze_benchmarks()
    feedback = get_feedback_summary()
    history = get_improvement_history(limit=5)

    # Derive trend from last cycles
    recent_scores = [
        c.get("benchmark_summary", {}).get("overall_avg_score")
        for c in history
        if c.get("benchmark_summary", {}).get("overall_avg_score") is not None
    ]
    trend = "improving" if len(recent_scores) >= 2 and recent_scores[0] > recent_scores[-1] else "stable"

    return {
        "benchmark": {
            "summary": analysis["summary"],
            "by_layer": analysis["by_layer"],
            "by_model": analysis["by_model"],
            "active_flags": len(analysis["flags"]),
        },
        "feedback": feedback,
        "improvement_cycles": {
            "recent_count": len(history),
            "latest": history[0] if history else None,
            "trend": trend,
        },
        "health": {
            "safety_failure_rate": analysis["summary"].get("safety_failure_rate", 0),
            "overall_score": analysis["summary"].get("overall_avg_score", 0),
            "feedback_avg_rating": feedback.get("avg_rating"),
        },
    }


@router.get("/health")
async def health():
    return {"status": "ok", "layer": "continuous_improvement"}
