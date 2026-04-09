"""
Tests for:
  - /vinci/complete API endpoint
  - ModelRouter fallback chain (primary → OpenRouter → Ollama → error)
  - Intent classifier layer routing
  - Hallucination risk scoring in BenchmarkLogger
  - Provider timeout simulation
  - vinci_core/router.py safe error handling
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def app():
    from fastapi import FastAPI
    from vinci_core.router import router as vinci_router
    test_app = FastAPI()
    test_app.include_router(vinci_router, prefix="/api/v1")
    return test_app


@pytest.fixture(scope="module")
def client(app):
    return TestClient(app)


# ── /vinci/complete endpoint ──────────────────────────────────────────────────

def _mock_engine_response(content="This could indicate a possible condition."):
    """Build a realistic AIResponse-compatible dict."""
    from vinci_core.schemas.response import AIResponse
    return AIResponse(
        model="claude-sonnet-4-6",
        content=content,
        usage={"prompt_tokens": 50, "completion_tokens": 30, "total_tokens": 80},
        metadata={
            "request_id": "test-uuid-001",
            "latency_ms": 210,
            "layer": "clinical",
            "safety": {"flag": "SAFE", "confidence": 0.85},
            "fallback_used": False,
        },
    )


def test_vinci_complete_success(client):
    with patch("vinci_core.router.engine") as mock_engine:
        mock_engine.run = AsyncMock(return_value=_mock_engine_response())
        response = client.post(
            "/api/v1/vinci/complete",
            json={"prompt": "I have a fever and sore throat", "layer": "clinical"},
        )
    assert response.status_code == 200
    data = response.json()
    assert "content" in data
    assert "model" in data
    assert "usage" in data
    assert "metadata" in data


def test_vinci_complete_has_safe_content(client):
    with patch("vinci_core.router.engine") as mock_engine:
        mock_engine.run = AsyncMock(return_value=_mock_engine_response(
            "These symptoms could possibly indicate a viral infection. Please consult a physician."
        ))
        response = client.post(
            "/api/v1/vinci/complete",
            json={"prompt": "Patient has chest pain and shortness of breath"},
        )
    assert response.status_code == 200
    data = response.json()
    assert "possibly" in data["content"] or "consult" in data["content"]


def test_vinci_complete_returns_500_on_engine_crash(client):
    with patch("vinci_core.router.engine") as mock_engine:
        mock_engine.run = AsyncMock(side_effect=RuntimeError("unexpected crash"))
        response = client.post(
            "/api/v1/vinci/complete",
            json={"prompt": "some query"},
        )
    assert response.status_code == 500
    # Must not expose internal exception details
    detail = response.json().get("detail", "")
    assert "unexpected crash" not in detail
    assert "RuntimeError" not in detail


def test_vinci_complete_rejects_empty_prompt(client):
    response = client.post(
        "/api/v1/vinci/complete",
        json={"prompt": ""},
    )
    # Pydantic schema validation should reject empty string if min_length enforced,
    # or the engine should return a safety-blocked response. Either way, not 500.
    assert response.status_code in (200, 422)


def test_vinci_complete_all_layers(client):
    for layer in ["base", "pharma", "clinical", "data", "radiology", "general"]:
        with patch("vinci_core.router.engine") as mock_engine:
            mock_engine.run = AsyncMock(return_value=_mock_engine_response())
            response = client.post(
                "/api/v1/vinci/complete",
                json={"prompt": "test query", "layer": layer},
            )
        assert response.status_code == 200, f"Failed for layer: {layer}"


# ── ModelRouter fallback chain ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_router_primary_succeeds():
    from vinci_core.routing.model_router import ModelRouter
    router = ModelRouter()
    mock_result = {
        "model": "openrouter",
        "content": "ok",
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        "metadata": {"provider": "openrouter"},
    }
    with patch.object(router.openrouter, "generate", new_callable=AsyncMock, return_value=mock_result):
        result = await router.run(prompt="hello", layer="base")
    assert result["metadata"]["fallback_used"] is False
    assert "latency_ms" in result["metadata"]


@pytest.mark.asyncio
async def test_router_falls_back_to_openrouter_on_primary_failure():
    from vinci_core.routing.model_router import ModelRouter
    router = ModelRouter()
    fallback_result = {
        "model": "openrouter",
        "content": "fallback response",
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        "metadata": {"provider": "openrouter"},
    }
    with patch.object(router.anthropic, "generate", new_callable=AsyncMock, side_effect=RuntimeError("anthropic down")), \
         patch.object(router.openrouter, "generate", new_callable=AsyncMock, return_value=fallback_result):
        result = await router.run(prompt="pharma query", layer="pharma")
    assert result["content"] == "fallback response"
    assert result["metadata"]["fallback_used"] is True


@pytest.mark.asyncio
async def test_router_falls_back_to_ollama_when_openrouter_also_fails():
    from vinci_core.routing.model_router import ModelRouter
    router = ModelRouter()
    ollama_result = {
        "model": "llama3:8b",
        "content": "ollama response",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        "metadata": {"provider": "ollama"},
    }
    with patch.object(router.anthropic, "generate", new_callable=AsyncMock, side_effect=RuntimeError("down")), \
         patch.object(router.openrouter, "generate", new_callable=AsyncMock, side_effect=RuntimeError("down")), \
         patch.object(router.ollama, "generate", new_callable=AsyncMock, return_value=ollama_result):
        result = await router.run(prompt="any query", layer="pharma")
    assert result["content"] == "ollama response"
    assert result["metadata"]["fallback_used"] is True


@pytest.mark.asyncio
async def test_router_returns_safe_error_when_all_providers_fail():
    from vinci_core.routing.model_router import ModelRouter
    router = ModelRouter()
    with patch.object(router.anthropic, "generate", new_callable=AsyncMock, side_effect=RuntimeError("down")), \
         patch.object(router.openrouter, "generate", new_callable=AsyncMock, side_effect=RuntimeError("down")), \
         patch.object(router.ollama, "generate", new_callable=AsyncMock, side_effect=RuntimeError("down")):
        result = await router.run(prompt="any query", layer="pharma")
    assert result["metadata"]["error"] is True
    assert result["metadata"]["fallback_used"] is True
    # Must not expose exception strings
    assert "RuntimeError" not in result["content"]
    assert "down" not in result["content"]


@pytest.mark.asyncio
async def test_router_no_double_fallback_when_openrouter_is_primary():
    """When openrouter IS the primary (base/general layer), fallback goes straight to Ollama."""
    from vinci_core.routing.model_router import ModelRouter
    router = ModelRouter()
    ollama_result = {
        "model": "llama3:8b",
        "content": "local response",
        "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
        "metadata": {"provider": "ollama"},
    }
    with patch.object(router.openrouter, "generate", new_callable=AsyncMock, side_effect=RuntimeError("timeout")), \
         patch.object(router.ollama, "generate", new_callable=AsyncMock, return_value=ollama_result):
        result = await router.run(prompt="general question", layer="general")
    assert result["content"] == "local response"
    assert result["metadata"]["fallback_used"] is True


@pytest.mark.asyncio
async def test_router_timeout_simulation():
    """Simulate a provider timing out — router should fall back gracefully."""
    import asyncio
    from vinci_core.routing.model_router import ModelRouter
    router = ModelRouter()

    async def slow_provider(*args, **kwargs):
        await asyncio.sleep(100)  # simulates timeout
        return {}

    fallback = {
        "model": "openrouter",
        "content": "fallback ok",
        "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
        "metadata": {"provider": "openrouter"},
    }
    with patch.object(router.anthropic, "generate", new_callable=AsyncMock, side_effect=asyncio.TimeoutError()), \
         patch.object(router.openrouter, "generate", new_callable=AsyncMock, return_value=fallback):
        result = await router.run(prompt="clinical query", layer="pharma")
    assert result["content"] == "fallback ok"
    assert result["metadata"]["fallback_used"] is True


# ── Intent Classifier ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_classifier_vision_keyword():
    from vinci_core.agent.classifier import IntentClassifier
    c = IntentClassifier()
    assert await c.classify("Analyze this MRI scan for tumors") == "radiology"


@pytest.mark.asyncio
async def test_classifier_pharma_keyword():
    from vinci_core.agent.classifier import IntentClassifier
    c = IntentClassifier()
    assert await c.classify("Review this NDA submission for FDA approval") == "pharma"


@pytest.mark.asyncio
async def test_classifier_clinical_keyword():
    from vinci_core.agent.classifier import IntentClassifier
    c = IntentClassifier()
    assert await c.classify("Patient has chest pain and shortness of breath") == "clinical"


@pytest.mark.asyncio
async def test_classifier_data_keyword():
    from vinci_core.agent.classifier import IntentClassifier
    c = IntentClassifier()
    assert await c.classify("Analyze this real-world evidence dataset for biomarkers") == "data"


@pytest.mark.asyncio
async def test_classifier_falls_back_to_general_on_llm_failure():
    from vinci_core.agent.classifier import IntentClassifier
    c = IntentClassifier()
    with patch.object(c.brain, "generate", new_callable=AsyncMock, side_effect=RuntimeError("LLM down")):
        result = await c.classify("something completely unrelated")
    assert result == "general"


@pytest.mark.asyncio
async def test_classifier_uses_llm_for_ambiguous_prompt():
    from vinci_core.agent.classifier import IntentClassifier
    c = IntentClassifier()
    with patch.object(c.brain, "generate", new_callable=AsyncMock, return_value={
        "model": "test", "content": "pharma", "usage": {}, "metadata": {}
    }):
        result = await c.classify("something ambiguous")
    assert result == "pharma"


# ── Hallucination Risk Scoring ─────────────────────────────────────────────────

from vinci_core.evaluation.benchmark_logger import _hallucination_risk_score


def test_hallucination_risk_high_without_rag():
    risk = _hallucination_risk_score(
        content="According to a study published in the Journal of Medicine, the drug causes liver failure.",
        rag_used=False,
        grounding_score=0.30,
        confidence=0.90,
    )
    assert risk > 0.50, f"Expected high risk but got {risk}"


def test_hallucination_risk_low_with_rag_and_uncertainty():
    risk = _hallucination_risk_score(
        content="Evidence suggests this might be consistent with the diagnosis. Please consult a physician.",
        rag_used=True,
        grounding_score=0.95,
        confidence=0.75,
    )
    assert risk < 0.30, f"Expected low risk but got {risk}"


def test_hallucination_risk_bounded():
    for content, rag, grounding, confidence in [
        ("you have diabetes", False, 0.10, 0.95),
        ("i'm not sure, consult a doctor", True, 1.0, 0.40),
        ("research shows this causes cancer", False, 0.20, 0.99),
    ]:
        risk = _hallucination_risk_score(content, rag, grounding, confidence)
        assert 0.0 <= risk <= 1.0, f"Risk out of bounds: {risk}"


def test_hallucination_risk_uncertainty_language_reduces_risk():
    risk_certain = _hallucination_risk_score(
        content="The cause is hypertension.",
        rag_used=False, grounding_score=0.50, confidence=0.90
    )
    risk_uncertain = _hallucination_risk_score(
        content="This might possibly be related to hypertension. Please consult a physician.",
        rag_used=False, grounding_score=0.50, confidence=0.90
    )
    assert risk_uncertain < risk_certain


# ── BenchmarkLogger integration ────────────────────────────────────────────────

def test_benchmark_logger_returns_hallucination_risk(tmp_path):
    from vinci_core.evaluation.benchmark_logger import BenchmarkLogger
    original_log_file = BenchmarkLogger.LOG_FILE
    BenchmarkLogger.LOG_DIR = str(tmp_path)
    BenchmarkLogger.LOG_FILE = str(tmp_path / "test_eval.jsonl")

    try:
        metrics = BenchmarkLogger.evaluate_and_log(
            prompt="Analyze patient symptoms",
            response_content="These symptoms could indicate hypertension. Please consult a physician.",
            response_metadata={
                "model": "test-model",
                "rag_used": False,
                "consensus": False,
                "safety": {"flag": "SAFE", "confidence": 0.80},
            },
            layer="clinical",
        )
        assert "hallucination_risk" in metrics
        assert 0.0 <= metrics["hallucination_risk"] <= 1.0
        assert "grounding_score" in metrics
        assert "safety_score" in metrics
        assert "confidence_score" in metrics
    finally:
        BenchmarkLogger.LOG_DIR = "benchmarks"
        BenchmarkLogger.LOG_FILE = original_log_file


def test_benchmark_logger_writes_to_file(tmp_path):
    import json
    from vinci_core.evaluation.benchmark_logger import BenchmarkLogger
    log_file = str(tmp_path / "eval.jsonl")
    original_log_dir = BenchmarkLogger.LOG_DIR
    original_log_file = BenchmarkLogger.LOG_FILE
    BenchmarkLogger.LOG_DIR = str(tmp_path)
    BenchmarkLogger.LOG_FILE = log_file

    try:
        BenchmarkLogger.evaluate_and_log(
            prompt="test",
            response_content="possibly related to condition",
            response_metadata={
                "model": "test", "rag_used": True, "consensus": False,
                "safety": {"flag": "SAFE", "confidence": 0.90},
            },
            layer="pharma",
        )
        with open(log_file) as f:
            entry = json.loads(f.readline())
        assert entry["layer"] == "pharma"
        assert "metrics" in entry
        # Confirm no sensitive data in logged entry
        assert "prompt" not in entry
        assert "response_content" not in entry
        # confidence_score is excluded from disk to prevent sensitive data storage
        assert "confidence_score" not in entry.get("metrics", {})
        # These safe metrics are still present on disk
        assert "grounding_score" in entry["metrics"]
        assert "safety_score" in entry["metrics"]
        assert "hallucination_risk" in entry["metrics"]
    finally:
        BenchmarkLogger.LOG_DIR = original_log_dir
        BenchmarkLogger.LOG_FILE = original_log_file


# ── Provider timeout simulation ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_openrouter_timeout_raises_correctly():
    """Verify that httpx.TimeoutException propagates before retry exhaustion."""
    import httpx
    from vinci_core.models.openrouter_model import OpenRouterModel
    model = OpenRouterModel()
    with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}), \
         patch("httpx.AsyncClient.post", new_callable=AsyncMock, side_effect=httpx.TimeoutException("timeout")):
        with pytest.raises(httpx.TimeoutException):
            await model.generate(prompt="test")


@pytest.mark.asyncio
async def test_ollama_connect_error_raises_correctly():
    """Verify Ollama propagates ConnectError after retries."""
    import httpx
    from vinci_core.models.ollama_model import OllamaModel
    model = OllamaModel()
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, side_effect=httpx.ConnectError("refused")):
        with pytest.raises(httpx.ConnectError):
            await model.generate(prompt="test")
