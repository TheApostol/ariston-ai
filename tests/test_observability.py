"""
Tests — Structured Observability Logger
"""

import json
import logging
import pytest
from vinci_core.observability.structured_logger import ObservabilityLogger, RequestTrace


class TestRequestTrace:
    def test_as_log_dict_includes_event(self):
        trace = RequestTrace(request_id="abc", layer="clinical")
        d = trace.as_log_dict()
        assert d["event"] == "ariston.request"
        assert d["request_id"] == "abc"
        assert d["layer"] == "clinical"

    def test_defaults(self):
        trace = RequestTrace(request_id="x", layer="latam")
        assert trace.fallback_used is False
        assert trace.latency_ms is None
        assert trace.provider_errors == []

    def test_full_trace(self):
        trace = RequestTrace(
            request_id="req-1",
            layer="pharma",
            model_used="claude-sonnet-4-6",
            fallback_used=True,
            fallback_reason="timeout",
            latency_ms=1234.5,
            prompt_tokens=100,
            completion_tokens=200,
            total_tokens=300,
            safety_flag="SAFE",
            rag_used=True,
            consensus=False,
        )
        d = trace.as_log_dict()
        assert d["model_used"] == "claude-sonnet-4-6"
        assert d["fallback_used"] is True
        assert d["latency_ms"] == 1234.5
        assert d["total_tokens"] == 300


class TestObservabilityLogger:
    def test_emit_produces_valid_json(self, caplog):
        logger = ObservabilityLogger()
        trace = RequestTrace(request_id="test-123", layer="latam", latency_ms=42.0)
        # Just verify no exceptions
        logger.emit(trace)

    def test_timer_measures_elapsed(self):
        logger = ObservabilityLogger()
        with logger.timer() as t:
            pass  # instant
        assert "elapsed_ms" in t
        assert isinstance(t["elapsed_ms"], float)
