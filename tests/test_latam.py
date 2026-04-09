"""
Tests — LATAM Regulatory Intelligence
Covers: agent, layer, pipeline, API endpoints, safety rules.
"""

import pytest
from vinci_core.agent.latam_agent import latam_agent, LATAM_AGENCIES
from vinci_core.layers.latam_layer import LatamLayer
from vinci_core.safety.guardrails import SafetyGuardrails, check_safety
from vinci_core.workflows.pipeline import Pipeline, PipelineContext, PipelineStep


# ---------------------------------------------------------------------------
# LatamRegulatoryAgent tests
# ---------------------------------------------------------------------------

class TestLatamAgent:
    def test_all_countries_returns_five_markets(self):
        countries = latam_agent.all_countries()
        assert set(countries) == {"brazil", "mexico", "colombia", "argentina", "chile"}

    def test_get_agency_profile_brazil(self):
        profile = latam_agent.get_agency_profile("brazil")
        assert profile["agency"] == "ANVISA"
        assert profile["language"] == "Portuguese"
        assert "RDC 204/2017" in profile["key_regulations"]

    def test_get_agency_profile_mexico(self):
        profile = latam_agent.get_agency_profile("mexico")
        assert profile["agency"] == "COFEPRIS"
        assert "NOM-177-SSA1" in profile["key_regulations"]

    def test_get_agency_profile_unknown_returns_empty(self):
        profile = latam_agent.get_agency_profile("unknown_country")
        assert profile == {}

    def test_case_insensitive_lookup(self):
        profile = latam_agent.get_agency_profile("BRAZIL")
        assert profile["agency"] == "ANVISA"

    def test_roadmap_all_countries(self):
        roadmap = latam_agent.build_multi_country_roadmap(
            countries=["brazil", "mexico", "colombia"],
        )
        assert "brazil" in roadmap["countries"]
        assert "mexico" in roadmap["countries"]
        assert "colombia" in roadmap["countries"]
        assert len(roadmap["recommended_sequence"]) == 3

    def test_roadmap_with_fda_approval_enables_expedited(self):
        roadmap = latam_agent.build_multi_country_roadmap(
            countries=["brazil"],
            has_fda_approval=True,
        )
        assert roadmap["countries"]["brazil"]["expedited_pathway_available"] is True

    def test_roadmap_without_reference_no_expedited(self):
        roadmap = latam_agent.build_multi_country_roadmap(
            countries=["argentina"],
            has_fda_approval=False,
            has_ema_approval=False,
        )
        assert roadmap["countries"]["argentina"]["expedited_pathway_available"] is False

    def test_roadmap_sequence_ordered_by_fastest_first(self):
        roadmap = latam_agent.build_multi_country_roadmap(
            countries=["brazil", "colombia"],
        )
        # Colombia (INVIMA) has faster review than Brazil (ANVISA)
        seq = roadmap["recommended_sequence"]
        assert seq[0] == "colombia"

    def test_submission_prompt_includes_agency(self):
        prompt = latam_agent.build_submission_prompt("mexico", "Novel antifungal tablet")
        assert "COFEPRIS" in prompt
        assert "MEXICO" in prompt

    def test_harmonization_context_contains_pandrh(self):
        context = latam_agent.format_harmonization_context()
        assert "PANDRH" in context
        assert "MERCOSUR" in context


# ---------------------------------------------------------------------------
# LatamLayer tests
# ---------------------------------------------------------------------------

class TestLatamLayer:
    def test_general_layer_builds_messages(self):
        layer = LatamLayer()
        messages = layer.build_messages("What are ANVISA requirements?")
        assert any(m["role"] == "system" for m in messages)
        assert any(m["role"] == "user" for m in messages)

    def test_brazil_layer_includes_anvisa_focus(self):
        layer = LatamLayer(country="brazil")
        assert "BRAZIL" in layer.system_prompt
        assert "ANVISA" in layer.system_prompt
        assert "SOLICITA" in layer.system_prompt

    def test_mexico_layer_includes_cofepris_focus(self):
        layer = LatamLayer(country="mexico")
        assert "COFEPRIS" in layer.system_prompt
        assert "FEUM" in layer.system_prompt

    def test_system_prompt_includes_uncertainty_disclaimer(self):
        layer = LatamLayer()
        lower = layer.system_prompt.lower()
        assert "never state that a product is approved" in lower or \
               "regulatory approval is determined by agencies" in lower


# ---------------------------------------------------------------------------
# Safety guardrails — LATAM-specific tests
# ---------------------------------------------------------------------------

class TestLatamSafetyGuardrails:
    def test_latam_approval_assertion_blocked(self):
        content = "ANVISA has approved this product for sale in Brazil."
        valid, _, meta = SafetyGuardrails.validate_output(content, layer="latam")
        assert not valid
        assert meta["safety_flag"] == "LATAM_APPROVAL_ASSERTION_BLOCKED"

    def test_latam_approval_cofepris_blocked(self):
        content = "COFEPRIS has approved the submission."
        valid, replaced, _ = SafetyGuardrails.validate_output(content, layer="latam")
        assert not valid
        assert "regulatory affairs professional" in replaced

    def test_safe_latam_output_passes(self):
        content = (
            "Based on ANVISA RDC 204/2017, the estimated review timeline is 180-365 days. "
            "We recommend consulting a regulatory affairs specialist before submission."
        )
        valid, _, meta = SafetyGuardrails.validate_output(content, layer="latam")
        assert valid
        assert meta["safety_flag"] == "SAFE"

    def test_clinical_definitive_still_blocked_in_latam_layer(self):
        content = "You have diabetes and must start insulin immediately."
        valid, _, meta = SafetyGuardrails.validate_output(content, layer="latam")
        assert not valid
        assert meta["safety_flag"] == "DEFINITIVE_DIAGNOSIS_BLOCKED"

    def test_check_safety_flags_latam_content(self):
        content = "The INVIMA registration requires submission via SIVICOS portal."
        result = check_safety(content, layer="latam")
        assert result["latam_regulatory_content"] is True

    def test_check_safety_requires_review_for_high_risk_without_uncertainty(self):
        content = "The dosage for this drug is 500mg twice daily."
        result = check_safety(content, layer="clinical")
        assert result["requires_review"] is True

    def test_check_safety_no_review_if_uncertain(self):
        content = "The dosage may be approximately 500mg, but further testing is recommended."
        result = check_safety(content, layer="clinical")
        assert result["requires_review"] is False


# ---------------------------------------------------------------------------
# Pipeline tests
# ---------------------------------------------------------------------------

class TestPipeline:
    @pytest.mark.asyncio
    async def test_pipeline_executes_steps_in_order(self):
        order = []

        async def step_a(ctx: PipelineContext) -> PipelineContext:
            order.append("a")
            ctx.results["a"] = True
            return ctx

        async def step_b(ctx: PipelineContext) -> PipelineContext:
            order.append("b")
            ctx.results["b"] = True
            return ctx

        pipeline = Pipeline(
            steps=[PipelineStep("a", step_a), PipelineStep("b", step_b)],
            name="test",
        )

        ctx = PipelineContext(prompt="test", layer="base")
        result = await pipeline.run(ctx)

        assert order == ["a", "b"]
        assert result.results["a"] is True
        assert result.results["b"] is True

    @pytest.mark.asyncio
    async def test_pipeline_captures_step_errors(self):
        async def failing_step(ctx: PipelineContext) -> PipelineContext:
            raise RuntimeError("step failed")

        pipeline = Pipeline(
            steps=[PipelineStep("fail", failing_step)],
            name="error_test",
        )
        ctx = PipelineContext(prompt="test", layer="base")
        result = await pipeline.run(ctx)

        assert len(result.errors) == 1
        assert "step failed" in result.errors[0]["error"]

    @pytest.mark.asyncio
    async def test_pipeline_aborts_on_flag(self):
        executed = []

        async def abort_step(ctx: PipelineContext) -> PipelineContext:
            ctx.metadata["abort_pipeline"] = True
            ctx.metadata["abort_reason"] = "safety_block"
            executed.append("abort")
            return ctx

        async def should_not_run(ctx: PipelineContext) -> PipelineContext:
            executed.append("should_not_run")
            return ctx

        pipeline = Pipeline(
            steps=[
                PipelineStep("abort", abort_step),
                PipelineStep("no_run", should_not_run),
            ],
            name="abort_test",
        )
        ctx = PipelineContext(prompt="test", layer="base")
        await pipeline.run(ctx)

        assert "abort" in executed
        assert "should_not_run" not in executed

    @pytest.mark.asyncio
    async def test_pipeline_sets_metadata(self):
        async def noop(ctx: PipelineContext) -> PipelineContext:
            return ctx

        pipeline = Pipeline(steps=[PipelineStep("noop", noop)], name="meta_test")
        ctx = PipelineContext(prompt="test", layer="latam")
        result = await pipeline.run(ctx)

        assert result.metadata["pipeline_name"] == "meta_test"
        assert "pipeline_latency_ms" in result.metadata


# ---------------------------------------------------------------------------
# LATAM Pipeline integration test (no engine call — country scoping + gap only)
# ---------------------------------------------------------------------------

class TestLatamPipelineSteps:
    @pytest.mark.asyncio
    async def test_country_scoping_detects_brazil(self):
        from vinci_core.workflows.latam_regulatory_pipeline import country_scoping_step
        ctx = PipelineContext(prompt="We need to register in Brazil via ANVISA.", layer="latam")
        result = await country_scoping_step(ctx)
        assert "brazil" in result.results["target_countries"]

    @pytest.mark.asyncio
    async def test_country_scoping_detects_multiple_markets(self):
        from vinci_core.workflows.latam_regulatory_pipeline import country_scoping_step
        ctx = PipelineContext(
            prompt="Registration plan for Mexico and Colombia markets.",
            layer="latam",
        )
        result = await country_scoping_step(ctx)
        countries = result.results["target_countries"]
        assert "mexico" in countries
        assert "colombia" in countries

    @pytest.mark.asyncio
    async def test_gap_analysis_detects_fda_approval(self):
        from vinci_core.workflows.latam_regulatory_pipeline import gap_analysis_step
        ctx = PipelineContext(prompt="Product has FDA approval and EMA clearance.", layer="latam")
        ctx.results["target_countries"] = ["brazil", "mexico"]
        result = await gap_analysis_step(ctx)
        assert result.results["reference_approvals"]["fda"] is True
        assert result.results["reference_approvals"]["ema"] is True
        assert result.metadata["expedited_eligible"] is True

    @pytest.mark.asyncio
    async def test_risk_assessment_appends_flags(self):
        from vinci_core.workflows.latam_regulatory_pipeline import risk_assessment_step
        ctx = PipelineContext(prompt="Brazil submission", layer="latam")
        ctx.results["target_countries"] = ["brazil"]
        ctx.results["roadmap"] = {
            "countries": {
                "brazil": {
                    "agency": "ANVISA",
                    "estimated_review_days": "54–219",
                }
            }
        }
        ctx.final_content = "Initial dossier strategy content."
        result = await risk_assessment_step(ctx)
        assert result.metadata["risk_count"] > 0
        assert any("ANVISA" in r for r in result.results["risk_flags"])
