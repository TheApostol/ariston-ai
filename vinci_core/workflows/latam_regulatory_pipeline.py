"""
LATAM Regulatory Pipeline — Ariston AI.

Composable 4-step pipeline for pharmaceutical regulatory submissions
targeting Latin American markets (ANVISA, COFEPRIS, INVIMA, ANMAT, ISP).

Steps:
  1. Country Scoping   — identify target markets and applicable agencies
  2. Gap Analysis      — compare existing approvals against LATAM requirements
  3. Dossier Strategy  — draft submission roadmap with per-country guidance
  4. Risk Assessment   — flag country-specific risks and mitigation paths

Usage:
    from vinci_core.workflows.latam_regulatory_pipeline import latam_pipeline
    ctx = PipelineContext(prompt="Submit Ozempic to Brazil and Mexico")
    result = await latam_pipeline.run(ctx)
"""

from __future__ import annotations

from vinci_core.workflows.pipeline import Pipeline, PipelineContext, PipelineStep, step
from vinci_core.agent.latam_agent import latam_agent


# ---------------------------------------------------------------------------
# Step 1: Country Scoping
# ---------------------------------------------------------------------------
@step("latam_country_scoping")
async def country_scoping_step(ctx: PipelineContext) -> PipelineContext:
    """Identify LATAM target markets from the prompt and build agency profiles."""
    prompt_lower = ctx.prompt.lower()

    country_keywords = {
        "brazil": ["brazil", "brasil", "anvisa"],
        "mexico": ["mexico", "méxico", "cofepris", "mex"],
        "colombia": ["colombia", "invima", "col"],
        "argentina": ["argentina", "anmat", "arg"],
        "chile": ["chile", "isp", "isp chile"],
    }

    detected = []
    for country, keywords in country_keywords.items():
        if any(kw in prompt_lower for kw in keywords):
            detected.append(country)

    if not detected:
        detected = list(latam_agent.all_countries())  # default: all LATAM

    profiles = {c: latam_agent.get_agency_profile(c) for c in detected}
    ctx.results["target_countries"] = detected
    ctx.results["agency_profiles"] = profiles
    ctx.metadata["latam_countries"] = detected

    return ctx


# ---------------------------------------------------------------------------
# Step 2: Reference Approval Gap Analysis
# ---------------------------------------------------------------------------
@step("latam_gap_analysis")
async def gap_analysis_step(ctx: PipelineContext) -> PipelineContext:
    """Determine expedited pathway eligibility based on existing reference approvals."""
    prompt_lower = ctx.prompt.lower()

    has_fda = any(kw in prompt_lower for kw in ["fda", "us approval", "us-approved"])
    has_ema = any(kw in prompt_lower for kw in ["ema", "ema-approved", "european"])
    has_who = any(kw in prompt_lower for kw in ["who prequalification", "who pq"])

    countries = ctx.results.get("target_countries", [])
    roadmap = latam_agent.build_multi_country_roadmap(
        countries=countries,
        has_fda_approval=has_fda,
        has_ema_approval=has_ema,
    )

    ctx.results["roadmap"] = roadmap
    ctx.results["reference_approvals"] = {
        "fda": has_fda,
        "ema": has_ema,
        "who_pq": has_who,
    }
    ctx.metadata["expedited_eligible"] = has_fda or has_ema or has_who

    return ctx


# ---------------------------------------------------------------------------
# Step 3: Dossier Strategy (engine call)
# ---------------------------------------------------------------------------
@step("latam_dossier_strategy")
async def dossier_strategy_step(ctx: PipelineContext) -> PipelineContext:
    """Generate per-country submission strategy using the LATAM engine layer."""
    from vinci_core.engine import engine

    countries = ctx.results.get("target_countries", [])
    roadmap = ctx.results.get("roadmap", {})
    reference_approvals = ctx.results.get("reference_approvals", {})

    approval_str = ", ".join(
        [k.upper() for k, v in reference_approvals.items() if v]
    ) or "None"

    recommended_seq = roadmap.get("recommended_sequence", countries)
    roadmap_summary = "\n".join(
        f"- {c.upper()} ({roadmap['countries'].get(c, {}).get('agency', '?')}): "
        f"est. {roadmap['countries'].get(c, {}).get('estimated_review_days', '?')} days review"
        for c in recommended_seq
    )

    strategy_prompt = (
        f"LATAM REGULATORY SUBMISSION STRATEGY\n"
        f"{'='*50}\n"
        f"Original Request: {ctx.prompt}\n\n"
        f"Target Countries (priority order): {', '.join(c.upper() for c in recommended_seq)}\n"
        f"Reference Approvals Held: {approval_str}\n\n"
        f"Registration Roadmap:\n{roadmap_summary}\n\n"
        f"Harmonization Context:\n{latam_agent.format_harmonization_context()}\n\n"
        f"Please provide a detailed dossier strategy covering:\n"
        f"1. Core Technical Dossier (CTD) modules required per country\n"
        f"2. Country-specific adaptations (labeling, language, local studies)\n"
        f"3. Simultaneous vs sequential submission strategy\n"
        f"4. Key regulatory contacts and pre-submission meeting recommendations\n"
        f"5. Estimated total cost range for full LATAM registration program"
    )

    response = await engine.run(
        prompt=strategy_prompt,
        layer="latam",
        use_rag=True,
    )

    ctx.results["dossier_strategy"] = response.content
    ctx.results["strategy_metadata"] = response.metadata
    ctx.final_content = response.content

    return ctx


# ---------------------------------------------------------------------------
# Step 4: Risk Assessment
# ---------------------------------------------------------------------------
@step("latam_risk_assessment")
async def risk_assessment_step(ctx: PipelineContext) -> PipelineContext:
    """Append country-specific risk flags to the pipeline output."""
    countries = ctx.results.get("target_countries", [])
    roadmap = ctx.results.get("roadmap", {})

    risks = []
    for country in countries:
        profile = roadmap.get("countries", {}).get(country, {})
        agency = profile.get("agency", country.upper())
        min_d, max_d = (
            profile.get("estimated_review_days", "?-?").split("–")
            if "–" in profile.get("estimated_review_days", "")
            else ("?", "?")
        )

        country_risks = {
            "brazil": [
                "ANVISA requires Portuguese-language dossier — translation costs apply",
                "VIGIMED post-marketing pharmacovigilance mandatory within 90 days of approval",
                "Local manufacturing or QP designee may be required for biologics",
            ],
            "mexico": [
                "COFEPRIS bioequivalence studies must use Mexican reference product (marca pionera)",
                "FEUM (Farmacopea) compliance required — may differ from USP/EP",
                "Review times extended by up to 6 months for new molecular entities",
            ],
            "colombia": [
                "INVIMA requires local regulatory representative (apoderado)",
                "Price regulation applies post-approval (Circular 03/2013 for price control)",
            ],
            "argentina": [
                "ANMAT requires local technical director (Director Técnico) — must be Argentine pharmacist",
                "Import registration requires AFIP tax compliance certification",
                "Price controls under Secretaría de Comercio may limit commercial viability",
            ],
            "chile": [
                "ISP requires bioequivalence studies per DS 3/2010 unless reference country recognized",
                "Import quota system may affect market access for certain products",
            ],
        }

        for risk in country_risks.get(country, []):
            risks.append(f"[{agency}] {risk}")

    ctx.results["risk_flags"] = risks
    ctx.metadata["risk_count"] = len(risks)

    if risks:
        risk_appendix = (
            "\n\n---\nRISK FLAGS\n"
            + "\n".join(f"⚠ {r}" for r in risks)
        )
        ctx.final_content += risk_appendix

    return ctx


# ---------------------------------------------------------------------------
# Assembled LATAM Regulatory Pipeline
# ---------------------------------------------------------------------------
latam_pipeline = Pipeline(
    steps=[
        country_scoping_step,
        gap_analysis_step,
        dossier_strategy_step,
        risk_assessment_step,
    ],
    name="latam_regulatory",
)
