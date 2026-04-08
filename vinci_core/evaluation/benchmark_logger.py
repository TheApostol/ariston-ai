import json
import os
from datetime import datetime
from typing import Dict, Any

class BenchmarkLogger:
    """
    MLCommons / MedPerf style benchmarking logger.
    Evaluates responses against clinical constraints and logs them.
    """
    LOG_DIR = "benchmarks"
    LOG_FILE = os.path.join(LOG_DIR, "eval_logs.jsonl")

    @classmethod
    def setup(cls):
        if not os.path.exists(cls.LOG_DIR):
            os.makedirs(cls.LOG_DIR)

    @classmethod
    def evaluate_and_log(cls, request_context: dict, response_metadata: dict, response_content: str):
        cls.setup()
        
        prompt = request_context.get("prompt", "")
        
        # 1. Evidence Grounding Eval
        grounded_score = 0.0
        if "PubMed Evidence:" in prompt or "RxNorm Classes:" in prompt:
            # Basic heuristic: does the model utilize/acknowledge the evidence?
            lower_content = response_content.lower()
            if any(term in lower_content for term in ["clinical", "data", "pubmed", "evidence", "classes", "literature"]):
                grounded_score = 0.95
            else:
                grounded_score = 0.50
        else:
            grounded_score = 1.0 

        # 2. Safety Eval
        safety_flag = response_metadata.get("safety_flag", "UNKNOWN")
        safety_score = 1.0 if safety_flag == "SAFE" else 0.0

        # Create rigorous ML Commons style event block
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "layer": request_context.get("layer", "unknown"),
            "model_provider": response_metadata.get("provider", "unknown"),
            "latency_ms": response_metadata.get("latency_ms", 0),
            "fallback_triggered": response_metadata.get("fallback_used", False),
            "metrics": {
                "grounding_score": grounded_score,
                "safety_score": safety_score,
                "confidence_score": response_metadata.get("confidence", 1.0)
            },
            "eval_flags": {
                "guardrail_status": safety_flag
            }
        }
        
        cls._write_log(log_entry)
        
        # Append to metadata so it's transparent to API caller
        response_metadata["benchmark_metrics"] = log_entry["metrics"]

    @classmethod
    def _write_log(cls, log_entry: Dict[str, Any]):
        try:
            with open(cls.LOG_FILE, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            print(f"Benchmark log failed: {e}")
