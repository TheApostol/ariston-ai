"""
Structured Observability Logger — Ariston AI.

Emits JSON-formatted log events for every request through the Vinci Engine.
Designed for production monitoring, debugging, and compliance audit trails.

Captures:
- request_id
- model_used
- fallback_used
- latency_ms
- provider_errors
- safety_flag
- layer
- token usage
"""

import json
import logging
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Optional


# Production-grade JSON logger
_json_logger = logging.getLogger("ariston.observability")
_json_logger.setLevel(logging.INFO)

if not _json_logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("%(message)s"))
    _json_logger.addHandler(_handler)
    _json_logger.propagate = False


@dataclass
class RequestTrace:
    request_id: str
    layer: str
    model_requested: Optional[str] = None
    model_used: Optional[str] = None
    fallback_used: bool = False
    fallback_reason: Optional[str] = None
    latency_ms: Optional[float] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    safety_flag: str = "SAFE"
    rag_used: bool = False
    consensus: bool = False
    provider_errors: list = field(default_factory=list)
    extra: dict = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def as_log_dict(self) -> dict:
        d = asdict(self)
        d["event"] = "ariston.request"
        return d


class ObservabilityLogger:
    """
    Central observability logger for Ariston AI.
    Call emit() after each engine run to produce a structured JSON log line.
    """

    def emit(self, trace: RequestTrace) -> None:
        _json_logger.info(json.dumps(trace.as_log_dict(), default=str))

    def emit_provider_error(
        self,
        request_id: str,
        provider: str,
        error: str,
        fallback_to: Optional[str] = None,
    ) -> None:
        record = {
            "event": "ariston.provider_error",
            "request_id": request_id,
            "provider": provider,
            "error": error,
            "fallback_to": fallback_to,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        _json_logger.warning(json.dumps(record))

    def emit_safety_event(
        self,
        request_id: str,
        flag: str,
        layer: str,
        prompt_preview: str,
    ) -> None:
        record = {
            "event": "ariston.safety_flag",
            "request_id": request_id,
            "flag": flag,
            "layer": layer,
            "prompt_preview": prompt_preview[:120],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        _json_logger.warning(json.dumps(record))

    @contextmanager
    def timer(self):
        """Context manager that yields a mutable dict with 'elapsed_ms'."""
        result = {}
        start = time.perf_counter()
        try:
            yield result
        finally:
            result["elapsed_ms"] = round((time.perf_counter() - start) * 1000, 2)


obs_logger = ObservabilityLogger()
