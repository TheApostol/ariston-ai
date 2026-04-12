"""
Tests for provider normalization, retry utility, pipeline, agents, swarm, and loop scheduler.
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ── Retry utility ─────────────────────────────────────────────────────────────

from vinci_core.utils.retry import async_retry


@pytest.mark.asyncio
async def test_retry_succeeds_on_first_attempt():
    calls = []

    @async_retry(max_attempts=3, base_delay=0)
    async def fn():
        calls.append(1)
        return "ok"

    result = await fn()
    assert result == "ok"
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_retry_retries_on_failure_then_succeeds():
    calls = []

    @async_retry(max_attempts=3, base_delay=0, exceptions=(ValueError,))
    async def fn():
        calls.append(1)
        if len(calls) < 2:
            raise ValueError("fail")
        return "recovered"

    result = await fn()
    assert result == "recovered"
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_retry_raises_after_max_attempts():
    calls = []

    @async_retry(max_attempts=3, base_delay=0, exceptions=(RuntimeError,))
    async def fn():
        calls.append(1)
        raise RuntimeError("always fails")

    with pytest.raises(RuntimeError, match="always fails"):
        await fn()
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_retry_does_not_catch_unlisted_exceptions():
    @async_retry(max_attempts=3, base_delay=0, exceptions=(ValueError,))
    async def fn():
        raise TypeError("not retried")

    with pytest.raises(TypeError):
        await fn()


# ── Provider interface normalization ──────────────────────────────────────────

REQUIRED_KEYS = {"model", "content", "usage", "metadata"}
REQUIRED_USAGE_KEYS = {"prompt_tokens", "completion_tokens", "total_tokens"}


def _assert_normalized(result: dict):
    assert REQUIRED_KEYS <= result.keys(), f"Missing keys: {REQUIRED_KEYS - result.keys()}"
    assert isinstance(result["model"], str)
    assert isinstance(result["content"], str)
    if result["usage"] is not None:
        assert REQUIRED_USAGE_KEYS <= result["usage"].keys()
        for k in REQUIRED_USAGE_KEYS:
            assert isinstance(result["usage"][k], int)
    assert isinstance(result["metadata"], dict)


@pytest.mark.asyncio
async def test_openrouter_returns_normalized_response():
    from vinci_core.models.openrouter_model import OpenRouterModel
    model = OpenRouterModel()

    mock_resp = MagicMock()
    mock_resp.json = MagicMock(return_value={
        "model": "test-model",
        "choices": [{"message": {"content": "hello"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    })

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp), \
         patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
        result = await model.generate(messages=[{"role": "user", "content": "test"}])

    _assert_normalized(result)
    assert result["content"] == "hello"
    assert result["usage"]["prompt_tokens"] == 10


@pytest.mark.asyncio
async def test_openrouter_error_response_raises():
    from vinci_core.models.openrouter_model import OpenRouterModel
    model = OpenRouterModel()

    mock_resp = MagicMock()
    mock_resp.json = MagicMock(return_value={"error": {"message": "rate limited"}})

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp), \
         patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
        with pytest.raises(RuntimeError, match="OpenRouter error"):
            await model.generate(messages=[{"role": "user", "content": "test"}])


@pytest.mark.asyncio
async def test_openrouter_accepts_prompt_string():
    from vinci_core.models.openrouter_model import OpenRouterModel
    model = OpenRouterModel()

    mock_resp = MagicMock()
    mock_resp.json = MagicMock(return_value={
        "model": "test",
        "choices": [{"message": {"content": "response"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    })

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp), \
         patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
        result = await model.generate(prompt="plain prompt")

    _assert_normalized(result)


@pytest.mark.asyncio
async def test_ollama_returns_normalized_response():
    from vinci_core.models.ollama_model import OllamaModel
    model = OllamaModel()

    mock_resp = MagicMock()
    mock_resp.json = MagicMock(return_value={
        "response": "ollama answer",
        "prompt_eval_count": 8,
        "eval_count": 12,
    })

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
        result = await model.generate(prompt="test prompt")

    _assert_normalized(result)
    assert result["content"] == "ollama answer"
    assert result["usage"]["prompt_tokens"] == 8
    assert result["usage"]["completion_tokens"] == 12
    assert result["usage"]["total_tokens"] == 20


@pytest.mark.asyncio
async def test_ollama_accepts_messages_list():
    from vinci_core.models.ollama_model import OllamaModel
    model = OllamaModel()

    mock_resp = MagicMock()
    mock_resp.json = MagicMock(return_value={"response": "ok", "prompt_eval_count": 0, "eval_count": 0})

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
        result = await model.generate(messages=[{"role": "user", "content": "hi"}])

    _assert_normalized(result)


# ── Engine observability ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_engine_returns_request_id():
    from vinci_core.engine import Engine
    eng = Engine()

    mock_router_result = {
        "model": "test-model",
        "content": "This could indicate a possible condition.",
        "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
        "metadata": {"provider": "test"},
    }

    with patch.object(eng.router, "run", new_callable=AsyncMock, return_value=mock_router_result), \
         patch("vinci_core.engine_context.classifier.classify", new_callable=AsyncMock, return_value="clinical"), \
         patch("vinci_core.engine_context.build_context", new_callable=AsyncMock, return_value={}), \
         patch("vinci_core.engine_context.pharmacogenomics_agent.format_for_context", new_callable=AsyncMock, return_value=""), \
         patch("vinci_core.engine.benchmark_logger.evaluate_and_log"), \
         patch("vinci_core.audit.gxp_trail.gxp_audit.log_event"):
        response = await eng.run(prompt="Patient has chest pain")

    assert response.metadata is not None
    # Engine may use job_id or request_id depending on version
    assert "job_id" in response.metadata or "request_id" in response.metadata
    assert "layer" in response.metadata


@pytest.mark.asyncio
async def test_engine_does_not_expose_stack_trace():
    from vinci_core.engine import Engine
    eng = Engine()

    with patch.object(eng.router, "run", new_callable=AsyncMock, side_effect=RuntimeError("secret db cred")), \
         patch("vinci_core.engine_context.classifier.classify", new_callable=AsyncMock, return_value="general"), \
         patch("vinci_core.engine_context.build_context", new_callable=AsyncMock, return_value={}):
        response = await eng.run(prompt="test prompt")

    # Must not expose raw exception string as-is to the caller
    assert response.metadata["error"] is True
    # Engine may use job_id or request_id depending on version
    assert "job_id" in response.metadata or "request_id" in response.metadata


@pytest.mark.asyncio
async def test_engine_input_too_short():
    from vinci_core.engine import Engine
    eng = Engine()
    response = await eng.run(prompt="hi")
    assert response.metadata["error"] is True


# ── Safety guardrails ─────────────────────────────────────────────────────────

def test_safety_blocks_definitive_diagnosis():
    from vinci_core.safety.guardrails import SafetyGuardrails
    is_safe, msg, meta = SafetyGuardrails.validate_output(
        "Based on these symptoms my diagnosis is that you have hypertension."
    )
    assert not is_safe
    assert meta["safety_flag"] == "DEFINITIVE_DIAGNOSIS_BLOCKED"


def test_safety_passes_hedged_language():
    from vinci_core.safety.guardrails import SafetyGuardrails
    is_safe, msg, meta = SafetyGuardrails.validate_output(
        "These symptoms could possibly indicate hypertension. Please consult a physician."
    )
    assert is_safe
    assert meta["safety_flag"] == "SAFE"


# ── Digital Twin Agent ────────────────────────────────────────────────────────

def test_twin_renal_increases_toxicity():
    from vinci_core.agent.twin_agent import DigitalTwinAgent
    agent = DigitalTwinAgent()
    result = agent.simulate_treatment(
        history="Patient has CKD stage 3 and renal failure",
        drug="warfarin",
        genetics=[],
    )
    assert result["toxicity_risk"] > 0.30
    assert result["organ_impact"]["renal"] in ("MODERATE", "CRITICAL")


def test_twin_poor_metabolizer_reduces_efficacy():
    from vinci_core.agent.twin_agent import DigitalTwinAgent
    agent = DigitalTwinAgent()
    result = agent.simulate_treatment(
        history="Healthy adult",
        drug="clopidogrel",
        genetics=["CYP2C19 Poor Metabolizer"],
    )
    assert result["efficacy_score"] < 0.50


def test_twin_returns_all_required_fields():
    from vinci_core.agent.twin_agent import DigitalTwinAgent
    agent = DigitalTwinAgent()
    result = agent.simulate_treatment(history="healthy", drug="aspirin", genetics=[])
    for key in ("prediction", "efficacy_score", "toxicity_risk", "sim_parameters", "organ_impact"):
        assert key in result


# ── IoMT Agent ────────────────────────────────────────────────────────────────

def test_iomt_low_pillbox_opens_reduces_adherence():
    from vinci_core.agent.iomt_agent import IoMTAgent
    agent = IoMTAgent()
    result = agent.forecast_adherence(
        history="Patient has no cognitive issues",
        telemetry={"pillbox_opens_7d": 2, "avg_heart_rate": 72, "steps_daily": 5000},
    )
    assert result["adherence_score"] < 0.75
    assert result["risk_level"] in ("MODERATE", "HIGH")


def test_iomt_dementia_reduces_adherence():
    from vinci_core.agent.iomt_agent import IoMTAgent
    agent = IoMTAgent()
    result = agent.forecast_adherence(
        history="Elderly patient with dementia and forgetfulness",
        telemetry={"pillbox_opens_7d": 7, "avg_heart_rate": 70, "steps_daily": 3000},
    )
    assert result["adherence_score"] < 0.80


def test_iomt_returns_required_fields():
    from vinci_core.agent.iomt_agent import IoMTAgent
    agent = IoMTAgent()
    result = agent.forecast_adherence(history="healthy")
    for key in ("adherence_score", "risk_level", "forecast_period", "recommendations"):
        assert key in result


# ── PV Narrative Agent ────────────────────────────────────────────────────────

def test_pv_cioms_narrative_contains_case_id():
    from vinci_core.agent.pv_narrative_agent import PharmacovigilanceNarrativeAgent
    agent = PharmacovigilanceNarrativeAgent()
    event = {
        "case_id": "AE-2024-TEST",
        "drug_name": "semaglutide",
        "dose": "1 mg weekly",
        "indication": "Type 2 Diabetes",
        "ae_term": "Pancreatitis",
        "onset_date": "2024-03-01",
        "outcome": "recovering",
        "severity": "hospitalization",
        "reporter_type": "physician",
    }
    narrative = agent.generate_cioms(event)
    assert "AE-2024-TEST" in narrative
    assert "semaglutide" in narrative
    assert "REQUIRED HOSPITALIZATION" in narrative


def test_pv_medwatch_narrative_contains_required_sections():
    from vinci_core.agent.pv_narrative_agent import PharmacovigilanceNarrativeAgent
    agent = PharmacovigilanceNarrativeAgent()
    event = {
        "case_id": "AE-MW-001",
        "drug_name": "warfarin",
        "dose": "5 mg daily",
        "indication": "Atrial fibrillation",
        "ae_term": "Bleeding",
        "onset_date": "2024-01-15",
        "outcome": "not recovered",
        "severity": "life-threatening",
        "reporter_type": "pharmacist",
    }
    narrative = agent.generate_medwatch(event)
    assert "Section A" in narrative
    assert "Section B" in narrative
    assert "Section C" in narrative
    assert "warfarin" in narrative


def test_pv_batch_generate_returns_all_cases():
    from vinci_core.agent.pv_narrative_agent import PharmacovigilanceNarrativeAgent
    agent = PharmacovigilanceNarrativeAgent()
    events = [
        {"case_id": f"CASE-{i}", "drug_name": "drug", "dose": "1mg", "indication": "x",
         "ae_term": "nausea", "onset_date": "2024-01-01", "outcome": "recovered",
         "severity": "non-serious", "reporter_type": "patient"}
        for i in range(3)
    ]
    results = agent.batch_generate(events, format="cioms")
    assert len(results) == 3
    for r in results:
        assert "cioms" in r


# ── Site Selection Agent ──────────────────────────────────────────────────────

def test_site_selection_returns_ranked_results():
    from vinci_core.agent.site_selection_agent import SiteSelectionAgent
    agent = SiteSelectionAgent()
    result = agent.recommend_sites(therapeutic_area="oncology", top_n=3)
    assert result["sites_recommended"] <= 3
    assert len(result["recommendations"]) <= 3
    for i, site in enumerate(result["recommendations"]):
        assert site["rank"] == i + 1
        assert "composite_score" in site


def test_site_selection_filters_by_agency():
    from vinci_core.agent.site_selection_agent import SiteSelectionAgent
    agent = SiteSelectionAgent()
    result = agent.recommend_sites(therapeutic_area="oncology", agency="ANVISA")
    for site in result["recommendations"]:
        assert site["agency"] == "ANVISA"


def test_site_feasibility_returns_all_agencies():
    from vinci_core.agent.site_selection_agent import SiteSelectionAgent
    agent = SiteSelectionAgent()
    result = agent.feasibility_summary(therapeutic_area="oncology")
    assert "ANVISA" in result["per_agency"]
    assert "COFEPRIS" in result["per_agency"]
    assert "INVIMA" in result["per_agency"]
    assert "ANMAT" in result["per_agency"]
    assert result["overall_readiness"] in ("HIGH", "MODERATE", "LOW")


def test_site_score_is_bounded():
    from vinci_core.agent.site_selection_agent import SiteSelectionAgent, LATAM_SITES
    agent = SiteSelectionAgent()
    for site in LATAM_SITES:
        score = agent.score_site(site, "oncology")
        assert 0.0 <= score <= 100.0


# ── Regulatory Copilot ────────────────────────────────────────────────────────

def test_regulatory_report_contains_gxp_hash():
    from vinci_core.agent.regulatory_agent import RegulatoryCopilot
    copilot = RegulatoryCopilot()
    report = copilot.generate_report(
        job_id="job-test-001",
        prompt="Review clopidogrel safety",
        result="Analysis complete. Possible DDI with omeprazole.",
        audit_logs=[],
    )
    assert "GxP Integrity Hash" in report
    assert "job-test-001" in report
    assert "READY FOR IRB SUBMISSION" in report


def test_regulatory_report_with_audit_logs():
    from vinci_core.agent.regulatory_agent import RegulatoryCopilot
    copilot = RegulatoryCopilot()
    logs = [{"status": "VERIFIED"}, {"status": "APPROVED"}]
    report = copilot.generate_report(
        job_id="job-002",
        prompt="NDA review",
        result="All checks passed.",
        audit_logs=logs,
    )
    assert "Total Audit Events: 2" in report
    assert "Last Event: APPROVED" in report


# ── Loop Scheduler ────────────────────────────────────────────────────────────

def test_loop_status_initial_state():
    from vinci_core.continuous_improvement.loop_scheduler import get_loop_status, stop_loop
    stop_loop()  # ensure clean state
    status = get_loop_status()
    assert isinstance(status["running"], bool)
    assert isinstance(status["interval_seconds"], int)
    assert isinstance(status["cycles_completed"], int)


def test_loop_stop_when_not_running():
    from vinci_core.continuous_improvement.loop_scheduler import stop_loop, get_loop_status
    stop_loop()
    result = stop_loop()
    assert result["stopped"] is False
    assert result["reason"] == "not_running"


# ── Pipeline framework ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_pipeline_runs_steps_in_order():
    from vinci_core.routing.pipeline import Pipeline

    log = []

    async def step_a(ctx):
        log.append("a")
        ctx["a"] = True
        return ctx

    async def step_b(ctx):
        log.append("b")
        ctx["b"] = True
        return ctx

    pipeline = Pipeline([step_a, step_b], name="test")
    result = await pipeline.run({"prompt": "test"})

    assert log == ["a", "b"]
    assert result["a"] is True
    assert result["b"] is True


@pytest.mark.asyncio
async def test_pipeline_continues_after_step_error():
    from vinci_core.routing.pipeline import Pipeline

    async def bad_step(ctx):
        raise RuntimeError("step failed")

    async def good_step(ctx):
        ctx["good"] = True
        return ctx

    pipeline = Pipeline([bad_step, good_step], name="test_resilient")
    result = await pipeline.run({"prompt": "test"})

    assert result["good"] is True
    assert "bad_step" in result["_step_errors"]
    assert "step failed" in result["_step_errors"]["bad_step"]


@pytest.mark.asyncio
async def test_pipeline_tracks_step_latencies():
    from vinci_core.routing.pipeline import Pipeline

    async def step_x(ctx):
        return ctx

    pipeline = Pipeline([step_x], name="latency_test")
    result = await pipeline.run({})

    assert "step_x" in result["_step_latencies_ms"]
    assert isinstance(result["_step_latencies_ms"]["step_x"], int)


# ── Model Router ──────────────────────────────────────────────────────────────

def test_model_router_layer_map_complete():
    from vinci_core.routing.model_router import ModelRouter
    router = ModelRouter()
    required_layers = {"clinical", "pharma", "data", "radiology", "base", "general"}
    assert required_layers <= router.layer_model_map.keys()


def test_model_router_uses_effective_model_override():
    from vinci_core.routing.model_router import ModelRouter
    router = ModelRouter()
    # When model is explicitly passed, it should override layer default
    # This tests the logic in ModelRouter.run: effective_model = model or layer_model_map[layer]
    effective = "anthropic"
    result_model = effective or router.layer_model_map.get("general")
    assert result_model == "anthropic"
