"""
Benchmark Analyzer — reads eval_logs.jsonl and surfaces low-scoring patterns.

Flags routing decisions with <75% success rate and generates
actionable insights for the improvement agent.
"""

import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from typing import List, Optional

EVAL_LOG = "benchmarks/eval_logs.jsonl"
SUCCESS_THRESHOLD = 0.75    # flag layers/models below this combined score
MIN_SAMPLES = 3             # minimum samples before flagging


def _load_logs(log_file: str = EVAL_LOG) -> List[dict]:
    """Load all benchmark log entries."""
    if not os.path.exists(log_file):
        return []
    entries = []
    try:
        with open(log_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    except Exception as e:
        print(f"[BenchmarkAnalyzer] Failed to load logs: {e}")
    return entries


def _combined_score(metrics: dict) -> float:
    """Weighted combined score: safety (50%) + grounding (30%) + confidence (20%)."""
    return (
        metrics.get("safety_score", 0.0) * 0.50
        + metrics.get("grounding_score", 0.0) * 0.30
        + metrics.get("confidence_score", 0.0) * 0.20
    )


def analyze_benchmarks(log_file: str = EVAL_LOG, threshold: float = SUCCESS_THRESHOLD) -> dict:
    """
    Analyze benchmark logs and return aggregated performance by layer and model.

    Returns:
        {
            "summary": {...},
            "by_layer": {layer: {avg_score, sample_count, flagged}},
            "by_model": {model: {avg_score, sample_count, flagged}},
            "flags": [{type, target, avg_score, sample_count, recommendation}],
            "analyzed_at": ISO timestamp,
        }
    """
    entries = _load_logs(log_file)
    if not entries:
        return {
            "summary": {"total_entries": 0, "flagged_count": 0},
            "by_layer": {},
            "by_model": {},
            "flags": [],
            "analyzed_at": datetime.now(timezone.utc).isoformat(),
        }

    # Aggregate by layer and model
    layer_scores: dict = defaultdict(list)
    model_scores: dict = defaultdict(list)
    safety_failures = 0

    for entry in entries:
        metrics = entry.get("metrics", {})
        score = _combined_score(metrics)
        layer = entry.get("layer", "unknown")
        model = entry.get("model", "unknown")

        layer_scores[layer].append(score)
        model_scores[model].append(score)

        if entry.get("safety_flag") != "SAFE":
            safety_failures += 1

    flags = []

    def _aggregate(score_map: dict, flag_type: str) -> dict:
        result = {}
        for key, scores in score_map.items():
            avg = sum(scores) / len(scores)
            flagged = len(scores) >= MIN_SAMPLES and avg < threshold
            result[key] = {
                "avg_score": round(avg, 3),
                "sample_count": len(scores),
                "flagged": flagged,
                "min_score": round(min(scores), 3),
                "max_score": round(max(scores), 3),
            }
            if flagged:
                flags.append({
                    "type": flag_type,
                    "target": key,
                    "avg_score": round(avg, 3),
                    "sample_count": len(scores),
                    "recommendation": _generate_recommendation(flag_type, key, avg),
                })
        return result

    by_layer = _aggregate(layer_scores, "layer")
    by_model = _aggregate(model_scores, "model")

    total = len(entries)
    return {
        "summary": {
            "total_entries": total,
            "flagged_count": len(flags),
            "overall_avg_score": round(
                sum(_combined_score(e.get("metrics", {})) for e in entries) / total, 3
            ) if total else 0.0,
            "safety_failure_count": safety_failures,
            "safety_failure_rate": round(safety_failures / total, 3) if total else 0.0,
            "threshold": threshold,
        },
        "by_layer": by_layer,
        "by_model": by_model,
        "flags": flags,
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
    }


def get_low_scoring_patterns(
    log_file: str = EVAL_LOG,
    threshold: float = SUCCESS_THRESHOLD,
    limit: int = 10,
) -> List[dict]:
    """
    Return the N most recent low-scoring log entries for targeted improvement.

    These are passed to the ImprovementAgent to generate specific fixes.
    """
    entries = _load_logs(log_file)
    low_scoring = [
        e for e in entries
        if _combined_score(e.get("metrics", {})) < threshold
    ]
    # Most recent first
    low_scoring.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
    return low_scoring[:limit]


def _generate_recommendation(flag_type: str, target: str, avg_score: float) -> str:
    """Generate a human-readable recommendation for a flagged pattern."""
    if flag_type == "layer":
        if avg_score < 0.5:
            return (
                f"Layer '{target}' is critically underperforming (score={avg_score:.2f}). "
                "Consider switching to a higher-capability model or adding domain-specific RAG context."
            )
        return (
            f"Layer '{target}' scoring below threshold (score={avg_score:.2f}). "
            "Review prompt engineering and RAG retrieval quality for this layer."
        )
    if flag_type == "model":
        if avg_score < 0.5:
            return (
                f"Model '{target}' is performing poorly (score={avg_score:.2f}). "
                "Consider removing from rotation or limiting to non-clinical layers."
            )
        return (
            f"Model '{target}' is below threshold (score={avg_score:.2f}). "
            "Consider routing clinical queries to a stronger model."
        )
    return f"Performance below threshold ({avg_score:.2f}). Review configuration."
