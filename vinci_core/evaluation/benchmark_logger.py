"""
MLCommons / MedPerf style benchmarking logger.
Evaluates every response against clinical constraints and logs to JSONL.
(Adapted from ariston-ai-1)
"""

import json
import os
from datetime import datetime, timezone
from typing import Dict, Any


class BenchmarkLogger:
    LOG_DIR = "benchmarks"
    LOG_FILE = os.path.join(LOG_DIR, "eval_logs.jsonl")

    @classmethod
    def _setup(cls):
        os.makedirs(cls.LOG_DIR, exist_ok=True)

    @classmethod
    def evaluate_and_log(
        cls,
        prompt: str,
        response_content: str,
        response_metadata: Dict[str, Any],
        layer: str = "unknown",
    ) -> Dict[str, Any]:
        cls._setup()

        # 1. Evidence grounding — did the model use retrieved knowledge?
        lower = response_content.lower()
        grounding_score = 0.0
        if any(k in prompt for k in ["Source", "PubMed", "FDA", "ClinicalTrials"]):
            grounding_score = 0.95 if any(
                t in lower for t in ["clinical", "evidence", "literature", "trial", "study"]
            ) else 0.50
        else:
            grounding_score = 0.75  # no RAG context, neutral

        # 2. Safety score
        safety = response_metadata.get("safety", {})
        safety_flag = safety.get("flag", "SAFE")
        safety_score = 1.0 if safety_flag == "SAFE" else 0.0

        # 3. Confidence from guardrails
        confidence = safety.get("confidence", 1.0)

        metrics = {
            "grounding_score": round(grounding_score, 3),
            "safety_score": round(safety_score, 3),
            "confidence_score": round(confidence, 3),
            "rag_used": response_metadata.get("rag_used", False),
            "consensus": response_metadata.get("consensus", False),
        }

        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "layer": layer,
            "model": response_metadata.get("model", "unknown"),
            "safety_flag": safety_flag,
            "metrics": metrics,
        }

        cls._write_log(log_entry)
        response_metadata["benchmark_metrics"] = metrics
        return metrics

    @classmethod
    def _write_log(cls, entry: Dict[str, Any]):
        # Whitelist only known-safe, non-sensitive fields before persisting.
        safe_entry = {
            "timestamp": entry.get("timestamp"),
            "layer": entry.get("layer"),
            "model": entry.get("model"),
            "safety_flag": entry.get("safety_flag"),
            "metrics": entry.get("metrics"),
        }
        try:
            with open(cls.LOG_FILE, "a") as f:
                f.write(json.dumps(safe_entry) + "\n")
        except Exception as e:
            print(f"[BenchmarkLogger] write failed: {type(e).__name__}")


benchmark_logger = BenchmarkLogger()
