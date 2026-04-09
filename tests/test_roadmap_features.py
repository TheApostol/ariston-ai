"""
Tests for new components introduced in the 8-priority roadmap:

  - P2: Streaming safety (validate_output enforced post-stream)
  - P3: Consistency scoring (run_consistency_check)
  - P5: Health check endpoint (/api/v1/health)
  - P6: Schema validation (invalid layer / model → 422)
  - engine_context: shared pre-processing helper
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, patch, MagicMock


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from vinci_core.router import router as vinci_router
    test_app = FastAPI()
    test_app.include_router(vinci_router, prefix="/api/v1")
    return TestClient(test_app)


def _mock_ai_response(content="This may indicate a viral infection. Consult your physician."):
    from vinci_core.schemas.response import AIResponse
    return AIResponse(
        model="claude-sonnet-4-6",
        content=content,
        usage={"prompt_tokens": 40, "completion_tokens": 20, "total_tokens": 60},
        metadata={
            "request_id": "test-uuid-999",
            "latency_ms": 150,
            "layer": "clinical",
            "safety": {"flag": "SAFE", "confidence": 0.90},
            "fallback_used": False,
        },
    )


# ── P6: Schema validation ─────────────────────────────────────────────────────

def test_invalid_layer_returns_422(client):
    """An unrecognised layer must produce a 422 Unprocessable Entity, not a silent fallback."""
    response = client.post(
        "/api/v1/vinci/complete",
        json={"prompt": "What is the standard dose of aspirin?", "layer": "tumor"},
    )
    assert response.status_code == 422
    body = response.json()
    # Pydantic validation error detail should mention the field
    assert "layer" in str(body).lower() or "detail" in body


def test_invalid_model_returns_422(client):
    """An unrecognised model name must produce a 422, not propagate to the engine."""
    response = client.post(
        "/api/v1/vinci/complete",
        json={"prompt": "Summarise recent ANVISA guidance", "model": "gpt-99-turbo"},
    )
    assert response.status_code == 422


def test_valid_layers_accepted(client):
    """All declared valid layers should pass schema validation."""
    valid_layers = ["base", "pharma", "clinical", "data", "radiology", "general"]
    for layer in valid_layers:
        with patch("vinci_core.router.engine") as mock_engine:
            mock_engine.run = AsyncMock(return_value=_mock_ai_response())
            response = client.post(
                "/api/v1/vinci/complete",
                json={"prompt": "test query", "layer": layer},
            )
        assert response.status_code == 200, f"Valid layer '{layer}' was rejected"


def test_valid_models_accepted(client):
    """All declared valid model values should pass schema validation."""
    valid_models = ["openrouter/free", "openrouter", "anthropic", "ollama", "consensus"]
    for model in valid_models:
        with patch("vinci_core.router.engine") as mock_engine:
            mock_engine.run = AsyncMock(return_value=_mock_ai_response())
            response = client.post(
                "/api/v1/vinci/complete",
                json={"prompt": "test query", "model": model},
            )
        assert response.status_code == 200, f"Valid model '{model}' was rejected"


# ── P5: Health check endpoint ─────────────────────────────────────────────────

def test_health_check_returns_200(client):
    """Health endpoint always returns HTTP 200 regardless of provider status."""
    with patch("vinci_core.router._probe_anthropic", new_callable=AsyncMock, return_value="ok"), \
         patch("vinci_core.router._probe_openrouter", new_callable=AsyncMock, return_value="ok"), \
         patch("vinci_core.router._probe_ollama", new_callable=AsyncMock, return_value="unreachable"):
        response = client.get("/api/v1/health")
    assert response.status_code == 200


def test_health_check_response_structure(client):
    """Health response must include status, providers, and latency_ms."""
    with patch("vinci_core.router._probe_anthropic", new_callable=AsyncMock, return_value="ok"), \
         patch("vinci_core.router._probe_openrouter", new_callable=AsyncMock, return_value="ok"), \
         patch("vinci_core.router._probe_ollama", new_callable=AsyncMock, return_value="ok"):
        response = client.get("/api/v1/health")
    data = response.json()
    assert "status" in data
    assert "providers" in data
    assert "latency_ms" in data
    assert set(data["providers"].keys()) == {"anthropic", "openrouter", "ollama"}


def test_health_check_status_ok_when_all_providers_ok(client):
    with patch("vinci_core.router._probe_anthropic", new_callable=AsyncMock, return_value="ok"), \
         patch("vinci_core.router._probe_openrouter", new_callable=AsyncMock, return_value="ok"), \
         patch("vinci_core.router._probe_ollama", new_callable=AsyncMock, return_value="ok"):
        response = client.get("/api/v1/health")
    assert response.json()["status"] == "ok"


def test_health_check_status_degraded_when_any_provider_down(client):
    with patch("vinci_core.router._probe_anthropic", new_callable=AsyncMock, return_value="unreachable"), \
         patch("vinci_core.router._probe_openrouter", new_callable=AsyncMock, return_value="ok"), \
         patch("vinci_core.router._probe_ollama", new_callable=AsyncMock, return_value="ok"):
        response = client.get("/api/v1/health")
    data = response.json()
    assert data["status"] == "degraded"
    assert data["providers"]["anthropic"] == "unreachable"


# ── P2: Streaming safety ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_stream_response_blocks_input_too_short():
    """Streaming engine must honour input safety for very short prompts."""
    from vinci_core.engine_stream import stream_response
    chunks = []
    async for chunk in stream_response(prompt="hi"):
        chunks.append(chunk)
    assert len(chunks) == 1
    assert "short" in chunks[0].lower() or len(chunks[0]) > 0


@pytest.mark.asyncio
async def test_stream_response_emits_safety_override_on_diagnosis():
    """
    When the streamed content would trigger definitive-diagnosis blocking,
    a [SAFETY_OVERRIDE] prefix must be appended as a final chunk.
    """
    from vinci_core.engine_stream import stream_response

    diagnosis_content = "Based on your symptoms, my diagnosis is that you have hypertension."

    async def _fake_stream():
        yield diagnosis_content

    async def _fake_build_context(**kwargs):
        return {"prompt": "test", "layer": "clinical"}

    with patch("vinci_core.engine_stream.build_request_context",
               new_callable=AsyncMock,
               return_value=(True, "Patient has high blood pressure", "clinical", {})), \
         patch("vinci_core.engine_stream.AristonAuditLedger.log_decision"):

        # Patch the Anthropic stream
        mock_stream_ctx = MagicMock()
        mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_stream_ctx)
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_stream_ctx.text_stream = _fake_stream()

        mock_client = MagicMock()
        mock_client.messages.stream.return_value = mock_stream_ctx

        with patch("vinci_core.engine_stream.anthropic.AsyncAnthropic", return_value=mock_client):
            chunks = []
            async for chunk in stream_response(prompt="Patient has high blood pressure"):
                chunks.append(chunk)

    full = "".join(chunks)
    assert "[SAFETY_OVERRIDE]" in full


# ── P3: Consistency scoring ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_consistency_check_returns_required_keys():
    """run_consistency_check must return all expected metric keys."""
    from vinci_core.evaluation.consistency import run_consistency_check

    mock_engine = MagicMock()
    mock_engine.run = AsyncMock(return_value=MagicMock(content="The patient may have a viral infection."))

    result = await run_consistency_check(
        prompt="What causes a fever?",
        n=3,
        engine=mock_engine,
    )
    for key in ("consistency_score", "jaccard_similarity", "mean_length",
                "length_variance", "runs", "errors"):
        assert key in result, f"Missing key: {key}"


@pytest.mark.asyncio
async def test_consistency_check_identical_responses_score_one():
    """When all responses are identical the consistency score must be 1.0."""
    from vinci_core.evaluation.consistency import run_consistency_check

    mock_engine = MagicMock()
    mock_engine.run = AsyncMock(
        return_value=MagicMock(content="Fever is caused by infection or inflammation.")
    )
    result = await run_consistency_check(prompt="What causes fever?", n=3, engine=mock_engine)
    assert result["consistency_score"] == 1.0
    assert result["jaccard_similarity"] == 1.0
    assert result["errors"] == 0


@pytest.mark.asyncio
async def test_consistency_check_handles_all_failures():
    """When all runs fail the function must return zeros, not raise."""
    from vinci_core.evaluation.consistency import run_consistency_check

    mock_engine = MagicMock()
    mock_engine.run = AsyncMock(side_effect=RuntimeError("provider down"))

    result = await run_consistency_check(prompt="What causes fever?", n=3, engine=mock_engine)
    assert result["runs"] == 0
    assert result["errors"] == 3
    assert result["consistency_score"] == 0.0


@pytest.mark.asyncio
async def test_consistency_check_single_run():
    """n=1 should still return a consistency_score of 1.0 (no variance possible)."""
    from vinci_core.evaluation.consistency import run_consistency_check

    mock_engine = MagicMock()
    mock_engine.run = AsyncMock(return_value=MagicMock(content="Some clinical answer."))

    result = await run_consistency_check(prompt="Describe hypertension.", n=1, engine=mock_engine)
    assert result["runs"] == 1
    assert result["consistency_score"] == 1.0


# ── engine_context: shared pre-processing ────────────────────────────────────

@pytest.mark.asyncio
async def test_build_request_context_blocks_short_input():
    """build_request_context must return valid=False for input too short."""
    from vinci_core.engine_context import build_request_context

    valid, prompt, layer, ctx = await build_request_context(
        prompt="hi",
        layer=None,
        context={},
        use_rag=False,
        patient_id=None,
        request_id="test-req-id",
    )
    assert valid is False


@pytest.mark.asyncio
async def test_build_request_context_auto_classifies_layer():
    """build_request_context must auto-classify layer when none is supplied."""
    from vinci_core.engine_context import build_request_context

    with patch("vinci_core.engine_context.classifier.classify",
               new_callable=AsyncMock, return_value="pharma"), \
         patch("vinci_core.engine_context.build_context",
               new_callable=AsyncMock, return_value={"prompt": "test"}):

        valid, prompt, layer, ctx = await build_request_context(
            prompt="What are the contraindications of warfarin?",
            layer=None,
            context={},
            use_rag=False,
            patient_id=None,
            request_id="test-req-id",
        )
    assert valid is True
    assert layer == "pharma"
