"""
Tests for Phase 3: Clinical Trial Intelligence + FDA 510(k) Preparation.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from vinci_core.workflows.pipeline import PipelineContext
from vinci_core.workflows.fda_510k_pipeline import (
    fda_510k_pipeline,
    predicate_search_step,
    substantial_equivalence_step,
    pccp_step,
    assemble_package_step,
    _KNOWN_AI_PREDICATES,
)
from vinci_core.workflows.clinical_trial_pipeline import (
    _LATAM_TRIAL_AUTHORITIES,
    patient_pool_step,
    regulatory_timeline_step,
    _risk_step_fixed,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_ctx(**kwargs) -> PipelineContext:
    return PipelineContext(
        prompt=kwargs.get("prompt", "test"),
        layer=kwargs.get("layer", "pharma"),
        metadata=kwargs.get("metadata", {}),
        results=kwargs.get("results", {}),
    )


# ---------------------------------------------------------------------------
# FDA 510(k) Pipeline — Unit Tests
# ---------------------------------------------------------------------------

class TestPredicateSearch:
    @pytest.mark.asyncio
    async def test_finds_radiology_match(self):
        ctx = make_ctx(metadata={"device_data": {"indication": "chest x-ray triage", "device_type": "radiology AI"}})
        result = await predicate_search_step(ctx)
        candidates = result.results["predicate_candidates"]
        assert len(candidates) > 0
        k_nums = [c["k_number"] for c in candidates]
        assert "K222016" in k_nums  # AI-Rad Companion

    @pytest.mark.asyncio
    async def test_finds_stroke_match(self):
        ctx = make_ctx(metadata={"device_data": {"indication": "large vessel occlusion stroke detection"}})
        result = await predicate_search_step(ctx)
        candidates = result.results["predicate_candidates"]
        k_nums = [c["k_number"] for c in candidates]
        assert "K213872" in k_nums or "K213322" in k_nums  # ContaCT or Viz.ai

    @pytest.mark.asyncio
    async def test_fallback_when_no_match(self):
        ctx = make_ctx(metadata={"device_data": {"indication": "completely unknown indication xyz", "device_type": "unknown"}})
        result = await predicate_search_step(ctx)
        candidates = result.results["predicate_candidates"]
        # Fallback: returns first 2 examples
        assert len(candidates) == 2

    @pytest.mark.asyncio
    async def test_indication_stored_in_results(self):
        ctx = make_ctx(metadata={"device_data": {"indication": "diabetic retinopathy screening"}})
        result = await predicate_search_step(ctx)
        assert result.results.get("indication") == "diabetic retinopathy screening"


class TestSubstantialEquivalence:
    @pytest.mark.asyncio
    async def test_se_argument_generated(self):
        ctx = make_ctx(results={
            "predicate_candidates": [{"k_number": "K190186", "device": "IDx-DR", "indication": "diabetic retinopathy screening"}],
            "indication": "retinal image analysis for diabetic retinopathy",
        })
        result = await substantial_equivalence_step(ctx)
        se = result.results["se_argument"]
        assert "K190186" in se
        assert "IDx-DR" in se
        assert "SUBSTANTIAL EQUIVALENCE" in se
        assert "513(i)" in se

    @pytest.mark.asyncio
    async def test_de_novo_when_no_predicate(self):
        ctx = make_ctx(results={"predicate_candidates": [], "indication": "novel AI device"})
        result = await substantial_equivalence_step(ctx)
        assert "De Novo" in result.results["se_argument"]


class TestPCCP:
    @pytest.mark.asyncio
    async def test_pccp_contains_required_sections(self):
        ctx = make_ctx()
        result = await pccp_step(ctx)
        pccp = result.results["pccp_draft"]
        assert "PREDETERMINED CHANGE CONTROL PLAN" in pccp
        assert "MODIFICATION PROTOCOL" in pccp
        assert "IMPACT ASSESSMENT" in pccp
        assert "PERFORMANCE MONITORING" in pccp
        assert "METHODOLOGY" in pccp

    @pytest.mark.asyncio
    async def test_pccp_threshold_specified(self):
        ctx = make_ctx()
        result = await pccp_step(ctx)
        assert "5%" in result.results["pccp_draft"]


class TestAssemblePackage:
    @pytest.mark.asyncio
    async def test_package_contains_all_sections(self):
        ctx = make_ctx(results={
            "predicate_candidates": [{"k_number": "K222016", "device": "AI-Rad Companion", "indication": "chest x-ray triage"}],
            "intended_use_draft": "AI-assisted chest X-ray triage tool.",
            "se_argument": "SE argument text here.",
            "pccp_draft": "PCCP content here.",
        })
        result = await assemble_package_step(ctx)
        pkg = result.final_content
        assert "SECTION 1" in pkg
        assert "SECTION 3" in pkg
        assert "SECTION 5" in pkg
        assert "SECTION 6" in pkg
        assert "SECTION 8" in pkg
        assert "K222016" in pkg

    @pytest.mark.asyncio
    async def test_package_includes_predicate_k_number(self):
        ctx = make_ctx(results={
            "predicate_candidates": [{"k_number": "K213872", "device": "ContaCT", "indication": "large vessel occlusion"}],
        })
        result = await assemble_package_step(ctx)
        assert "K213872" in result.final_content


class TestKnownPredicates:
    def test_predicate_list_not_empty(self):
        assert len(_KNOWN_AI_PREDICATES) >= 7

    def test_predicates_have_required_fields(self):
        for p in _KNOWN_AI_PREDICATES:
            assert "k_number" in p
            assert "device" in p
            assert "indication" in p

    def test_k_numbers_have_correct_format(self):
        for p in _KNOWN_AI_PREDICATES:
            assert p["k_number"].startswith("K")
            assert len(p["k_number"]) == 7


# ---------------------------------------------------------------------------
# FDA 510(k) Full Pipeline Integration
# ---------------------------------------------------------------------------

class TestFDA510kPipeline:
    @pytest.mark.asyncio
    async def test_pipeline_runs_without_ai_steps(self):
        """Test pipeline steps that don't require engine calls."""
        ctx = make_ctx(metadata={
            "device_data": {
                "device_name": "AI Chest X-ray Triage",
                "indication": "chest x-ray triage",
                "device_type": "radiology AI",
            }
        })
        # Run predicate search (no AI), SE, PCCP, assemble — skip intended_use (needs engine)
        ctx = await predicate_search_step(ctx)
        assert ctx.results.get("predicate_candidates")
        ctx = await substantial_equivalence_step(ctx)
        assert ctx.results.get("se_argument")
        ctx = await pccp_step(ctx)
        assert ctx.results.get("pccp_draft")
        ctx = await assemble_package_step(ctx)
        assert ctx.final_content
        assert "510(k)" in ctx.final_content


# ---------------------------------------------------------------------------
# Clinical Trial Pipeline — Unit Tests
# ---------------------------------------------------------------------------

class TestPatientPool:
    @pytest.mark.asyncio
    async def test_oncology_pool_brazil(self):
        ctx = make_ctx(results={
            "target_countries": ["brazil"],
            "indication": "oncology",
            "target_n": 300,
        })
        result = await patient_pool_step(ctx)
        pool = result.results["patient_pool"]
        assert "brazil" in pool
        assert pool["brazil"]["estimated_eligible_k"] == 500
        assert pool["brazil"]["estimated_screenable"] > 0

    @pytest.mark.asyncio
    async def test_feasibility_high_when_large_pool(self):
        ctx = make_ctx(results={
            "target_countries": ["brazil", "mexico"],
            "indication": "cardiovascular",
            "target_n": 100,
        })
        result = await patient_pool_step(ctx)
        assert result.results["enrollment_feasibility"] == "HIGH"

    @pytest.mark.asyncio
    async def test_feasibility_low_when_small_pool(self):
        ctx = make_ctx(results={
            "target_countries": ["chile"],
            "indication": "oncology",
            "target_n": 50000,
        })
        result = await patient_pool_step(ctx)
        assert result.results["enrollment_feasibility"] == "LOW"

    @pytest.mark.asyncio
    async def test_fallback_indication_matching(self):
        ctx = make_ctx(results={
            "target_countries": ["brazil"],
            "indication": "unknown_rare_disease",
            "target_n": 10,
        })
        result = await patient_pool_step(ctx)
        # Should fallback to oncology
        assert result.results.get("patient_pool") is not None


class TestRegulatoryTimeline:
    @pytest.mark.asyncio
    async def test_brazil_timeline(self):
        ctx = make_ctx(results={"target_countries": ["brazil"]})
        result = await regulatory_timeline_step(ctx)
        tl = result.results["regulatory_timeline"]
        assert "brazil" in tl
        assert tl["brazil"]["authority"] == "ANVISA + CONEP"
        assert tl["brazil"]["authorization_days"] == 90

    @pytest.mark.asyncio
    async def test_mexico_longest_auth(self):
        ctx = make_ctx(results={"target_countries": ["brazil", "mexico", "colombia"]})
        result = await regulatory_timeline_step(ctx)
        assert result.results["longest_auth_days"] == 120  # Mexico = 120 days

    @pytest.mark.asyncio
    async def test_all_countries_have_parallel_ethics(self):
        ctx = make_ctx(results={"target_countries": list(_LATAM_TRIAL_AUTHORITIES.keys())})
        result = await regulatory_timeline_step(ctx)
        for country, info in result.results["regulatory_timeline"].items():
            assert info["parallel_ethics_review"] is True


class TestRiskAssessment:
    @pytest.mark.asyncio
    async def test_brazil_risks_present(self):
        ctx = make_ctx(results={
            "target_countries": ["brazil"],
            "indication": "oncology",
            "phase": "II/III",
            "target_n": 300,
            "patient_pool": {"brazil": {"estimated_eligible_k": 500, "screening_rate_pct": 2.5, "estimated_screenable": 12500}},
            "recommended_sites": [],
            "regulatory_timeline": {"brazil": {"authority": "ANVISA + CONEP", "authorization_days": 90}},
            "enrollment_feasibility": "HIGH",
            "total_screenable": 12500,
            "protocol_analysis": "Protocol analysis summary.",
        })
        result = await _risk_step_fixed(ctx)
        risks = result.results["risk_flags"]
        brazil_risks = [r for r in risks if r.startswith("[BRAZIL]")]
        assert len(brazil_risks) > 0
        assert any("CONEP" in r for r in brazil_risks)

    @pytest.mark.asyncio
    async def test_argentina_risks_include_currency(self):
        ctx = make_ctx(results={
            "target_countries": ["argentina"],
            "indication": "cardiovascular",
            "phase": "III",
            "target_n": 200,
            "patient_pool": {},
            "recommended_sites": [],
            "regulatory_timeline": {},
            "enrollment_feasibility": "MODERATE",
            "total_screenable": 1000,
            "protocol_analysis": "",
        })
        result = await _risk_step_fixed(ctx)
        risks = result.results["risk_flags"]
        arg_risks = [r for r in risks if r.startswith("[ARGENTINA]")]
        assert any("instability" in r.lower() or "budget" in r.lower() for r in arg_risks)

    @pytest.mark.asyncio
    async def test_final_content_assembled(self):
        ctx = make_ctx(results={
            "target_countries": ["brazil", "mexico"],
            "indication": "oncology",
            "phase": "II",
            "target_n": 150,
            "patient_pool": {},
            "recommended_sites": [{"site": "São Paulo Hospital"}],
            "regulatory_timeline": {
                "brazil": {"authority": "ANVISA + CONEP", "authorization_days": 90},
                "mexico": {"authority": "COFEPRIS + CONBIOÉTICA", "authorization_days": 120},
            },
            "enrollment_feasibility": "HIGH",
            "total_screenable": 5000,
            "protocol_analysis": "Feasibility analysis here.",
        })
        result = await _risk_step_fixed(ctx)
        assert result.final_content is not None
        assert "CLINICAL TRIAL INTELLIGENCE SUMMARY" in result.final_content
        assert "RISK FLAGS" in result.final_content


class TestLatamTrialAuthorities:
    def test_all_five_countries_covered(self):
        expected = {"brazil", "mexico", "colombia", "argentina", "chile"}
        assert expected == set(_LATAM_TRIAL_AUTHORITIES.keys())

    def test_brazil_authority_details(self):
        brazil = _LATAM_TRIAL_AUTHORITIES["brazil"]
        assert "ANVISA" in brazil["body"]
        assert "CONEP" in brazil["body"]
        assert brazil["timeline_days"] == 90

    def test_colombia_fastest_auth(self):
        timelines = {c: v["timeline_days"] for c, v in _LATAM_TRIAL_AUTHORITIES.items()}
        assert timelines["colombia"] == min(timelines.values())


# ---------------------------------------------------------------------------
# Phase 3 API Endpoint Tests
# ---------------------------------------------------------------------------

class TestPhase3API:
    def test_trial_authorities_all_countries(self):
        """Verify authority data is accessible for all LATAM countries."""
        for country in ["brazil", "mexico", "colombia", "argentina", "chile"]:
            auth = _LATAM_TRIAL_AUTHORITIES.get(country)
            assert auth is not None
            assert "body" in auth
            assert "timeline_days" in auth

    def test_predicates_endpoint_data(self):
        """Verify predicate data structure for API endpoint."""
        for p in _KNOWN_AI_PREDICATES:
            assert all(k in p for k in ["k_number", "device", "indication"])

    def test_predicate_filtering_by_indication(self):
        """Simulate indication-based predicate filtering."""
        indication = "retinopathy"
        keywords = indication.lower().split()[:4]
        filtered = [
            p for p in _KNOWN_AI_PREDICATES
            if any(kw in p["indication"] for kw in keywords)
            or any(kw in p["device"].lower() for kw in keywords)
        ]
        assert len(filtered) > 0
        assert any("IDx-DR" in p["device"] for p in filtered)
