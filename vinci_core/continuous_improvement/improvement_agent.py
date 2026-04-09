"""
Improvement Agent — analyzes low-scoring patterns and generates
routing/model selection improvements using the Vinci Engine.

The agent:
  1. Reads flagged patterns from BenchmarkAnalyzer
  2. Reads unprocessed feedback signals
  3. Generates an improvement plan (JSON) via LLM
  4. Logs the plan for human review and optional auto-deployment
"""

import json
import os
from datetime import datetime, timezone
from typing import List, Optional

from vinci_core.engine import engine
from .benchmark_analyzer import analyze_benchmarks, get_low_scoring_patterns
from .feedback_loop import get_unprocessed_signals, mark_signal_processed

IMPROVEMENT_LOG = "benchmarks/improvement_log.jsonl"


def _write_log(entry: dict):
    os.makedirs("benchmarks", exist_ok=True)
    try:
        with open(IMPROVEMENT_LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        print(f"[ImprovementAgent] Log write failed: {e}")


def _build_improvement_prompt(
    analysis: dict,
    low_scoring_samples: List[dict],
    feedback_signals: List[dict],
) -> str:
    flags_text = json.dumps(analysis.get("flags", []), indent=2)
    samples_text = json.dumps(low_scoring_samples[:5], indent=2)
    signals_text = json.dumps(feedback_signals[:5], indent=2)

    return f"""You are the autonomous improvement agent for the Ariston AI Life Sciences platform.

Your task: Analyze performance data and generate a JSON improvement plan.

## BENCHMARK ANALYSIS SUMMARY
Overall avg score: {analysis['summary'].get('overall_avg_score', 'N/A')}
Flagged patterns: {len(analysis.get('flags', []))}
Safety failures: {analysis['summary'].get('safety_failure_count', 0)}
Threshold: {analysis['summary'].get('threshold', 0.75)}

## FLAGGED PATTERNS
{flags_text}

## LOW-SCORING SAMPLES (recent)
{samples_text}

## CUSTOMER FEEDBACK SIGNALS
{signals_text}

## INSTRUCTIONS
Generate a JSON improvement plan with this exact structure:
{{
  "priority": "high|medium|low",
  "improvements": [
    {{
      "id": "improvement-001",
      "type": "routing|prompt|model_selection|rag|safety",
      "target": "layer or model name",
      "issue": "description of the identified problem",
      "proposed_change": "specific change to make",
      "expected_impact": "expected improvement in score or metric",
      "risk_level": "low|medium|high",
      "requires_human_review": true|false
    }}
  ],
  "estimated_cost_reduction_pct": 0-100,
  "estimated_latency_improvement_ms": 0-5000,
  "summary": "brief executive summary"
}}

Focus on:
- Routing improvements (which model handles which layer)
- Prompt engineering improvements for low-scoring layers
- RAG retrieval quality improvements
- Safety guardrail adjustments for false positives
- Cost optimization (prefer cheaper models for low-complexity queries)

Only output valid JSON. No markdown fences, no explanations outside JSON."""


async def run_improvement_cycle(
    auto_process_signals: bool = True,
) -> dict:
    """
    Run one full improvement cycle:
      1. Analyze benchmarks
      2. Get low-scoring patterns
      3. Get feedback signals
      4. Generate improvement plan
      5. Log plan for review
      6. Mark signals as processed

    Returns the improvement plan dict.
    """
    # 1. Analyze
    analysis = analyze_benchmarks()
    low_scoring = get_low_scoring_patterns(limit=10)
    signals = get_unprocessed_signals()

    # If nothing to improve, return early
    if not analysis["flags"] and not signals:
        result = {
            "status": "no_action_needed",
            "message": "No low-scoring patterns or feedback signals detected",
            "analyzed_at": datetime.now(timezone.utc).isoformat(),
            "summary": analysis["summary"],
        }
        _write_log(result)
        return result

    # 2. Build prompt
    prompt = _build_improvement_prompt(analysis, low_scoring, signals)

    # 3. Generate improvement plan
    response = await engine.run(
        prompt=prompt,
        layer="general",
        use_rag=False,
    )

    # 4. Parse plan
    plan = {}
    try:
        # Extract JSON from response
        content = response.content.strip()
        # Handle potential markdown fences
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        plan = json.loads(content)
    except Exception as e:
        plan = {
            "error": f"Failed to parse improvement plan: {e}",
            "raw_response": response.content[:500],
        }

    # 5. Log the plan
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cycle_type": "automated",
        "benchmark_summary": analysis["summary"],
        "flags_count": len(analysis["flags"]),
        "signals_count": len(signals),
        "plan": plan,
        "model": response.model,
    }
    _write_log(log_entry)

    # 6. Mark signals processed
    if auto_process_signals:
        for signal in signals:
            try:
                mark_signal_processed(signal["id"])
            except Exception as e:
                print(f"[ImprovementAgent] Failed to mark signal {signal.get('id')}: {e}")

    return {
        "status": "plan_generated",
        "plan": plan,
        "benchmark_summary": analysis["summary"],
        "flags": analysis["flags"],
        "signals_processed": len(signals),
        "logged_to": IMPROVEMENT_LOG,
        "model": response.model,
    }


def get_improvement_history(limit: int = 20) -> List[dict]:
    """Load recent improvement cycle logs."""
    if not os.path.exists(IMPROVEMENT_LOG):
        return []
    entries = []
    try:
        with open(IMPROVEMENT_LOG) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    except Exception as e:
        print(f"[ImprovementAgent] Failed to load history: {e}")
        return []
    # Most recent first
    return list(reversed(entries))[-limit:]
