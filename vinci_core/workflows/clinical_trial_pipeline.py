"""
Clinical Trial Intelligence Pipeline — Phase 1→2 / Ariston AI LATAM.

AI-powered clinical trial optimization for LATAM markets:
  1. Protocol Analysis    — evaluate feasibility, site requirements, patient criteria
  2. Site Recommendation — score and rank LATAM sites (uses SiteSelectionAgent)
  3. Patient Matching     — eligibility criteria extraction + patient pool sizing
  4. Trial Timeline       — milestone planning per LATAM regulatory requirements
  5. Risk Assessment      — country-specific trial risks and mitigation

Target buyer: Sponsor companies and CROs running Phase II–III trials in LATAM.
Revenue: $100K–$500K per customer per trial (per Execution Roadmap Phase 1 target).

LATAM clinical trial context:
  - Brazil: CONEP ethics board + ANVISA clinical trial authorization (RDC 204/2017)
  - Mexico: COFEPRIS clinical trial authorization + CONBIOÉTICA ethics
  - Colombia: INVIMA clinical trial authorization + ethics committee
  - Argentina: ANMAT clinical trial authorization (Disposición 6677/2010)
  - Chile: ISP authorization + ethics committee per DS 114/2010
"""

from __future__ import annotations

from typing import Optional
from vinci_core.workflows.pipeline import Pipeline, PipelineContext, PipelineStep, step
from vinci_core.agent.site_selection_agent import SiteSelectionAgent

_site_agent = SiteSelectionAgent()

_LATAM_TRIAL_AUTHORITIES = {
    "brazil":    {"body": "ANVISA + CONEP", "timeline_days": 90,  "system": "ICTQ/Plataforma Brasil"},
    "mexico":    {"body": "COFEPRIS + CONBIOÉTICA", "timeline_days": 120, "system": "COFEPRIS portal"},
    "colombia":  {"body": "INVIMA + Comité de Ética", "timeline_days": 60,  "system": "INVIMA SIVICOS"},
    "argentina": {"body": "ANMAT + ethics committee", "timeline_days": 90,  "system": "ANMAT SAID"},
    "chile":     {"body": "ISP + ethics committee", "timeline_days": 75,  "system": "ISP portal"},
}


# ---------------------------------------------------------------------------
# Step 1: Protocol Analysis
# ---------------------------------------------------------------------------
@step("trial_protocol_analysis")
async def protocol_analysis_step(ctx: PipelineContext) -> PipelineContext:
    """Analyze protocol feasibility and extract key requirements."""
    from vinci_core.engine import engine

    protocol_data = ctx.metadata.get("protocol_data") or {}
    phase = protocol_data.get("phase", "II/III")
    indication = protocol_data.get("indication", "oncology")
    arm_count = protocol_data.get("arms", 2)
    target_n = protocol_data.get("target_enrollment", 300)
    duration_months = protocol_data.get("duration_months", 24)
    countries = protocol_data.get("countries", ["brazil", "mexico", "colombia"])

    analysis_prompt = (
        f"CLINICAL TRIAL PROTOCOL FEASIBILITY ANALYSIS\n"
        f"Phase: {phase} | Indication: {indication}\n"
        f"Arms: {arm_count} | Target N: {target_n} | Duration: {duration_months}m\n"
        f"LATAM Countries: {', '.join(c.upper() for c in countries)}\n"
        f"Query: {ctx.prompt}\n\n"
        f"Provide:\n"
        f"1. Feasibility assessment for LATAM enrollment\n"
        f"2. Key protocol complexity factors\n"
        f"3. Inclusion/exclusion criteria risk factors for LATAM patient populations\n"
        f"4. Biomarker or genetic screening requirements\n"
        f"5. Regulatory authorization timeline per country\n"
        f"6. Estimated patient recruitment rate per site per month in LATAM\n"
    )

    response = await engine.run(prompt=analysis_prompt, layer="pharma", use_rag=True)

    ctx.results["protocol_analysis"] = response.content
    ctx.results["target_countries"]  = countries
    ctx.results["target_n"]          = target_n
    ctx.results["indication"]        = indication
    ctx.results["phase"]             = phase
    return ctx


# ---------------------------------------------------------------------------
# Step 2: Site Recommendation
# ---------------------------------------------------------------------------
@step("trial_site_recommendation")
async def site_recommendation_step(ctx: PipelineContext) -> PipelineContext:
    """Score and rank LATAM clinical trial sites."""
    countries = ctx.results.get("target_countries", ["brazil", "mexico"])
    indication = ctx.results.get("indication", "oncology")
    target_n = ctx.results.get("target_n", 300)

    sites = _site_agent.recommend_sites(
        therapeutic_area=indication,
        target_countries=countries,
        min_sites=max(3, target_n // 50),
    )

    ctx.results["recommended_sites"] = sites
    ctx.results["site_count"] = len(sites)
    return ctx


# ---------------------------------------------------------------------------
# Step 3: Patient Pool Sizing
# ---------------------------------------------------------------------------
@step("trial_patient_pool")
async def patient_pool_step(ctx: PipelineContext) -> PipelineContext:
    """Estimate patient pool size across LATAM countries."""
    countries = ctx.results.get("target_countries", [])
    indication = ctx.results.get("indication", "")
    target_n = ctx.results.get("target_n", 300)

    # Simplified prevalence estimates per indication per LATAM country (thousands)
    _prevalence_k = {
        "oncology":         {"brazil": 500, "mexico": 300, "colombia": 150, "argentina": 100, "chile": 50},
        "cardiovascular":   {"brazil": 8000, "mexico": 5000, "colombia": 2000, "argentina": 2000, "chile": 800},
        "type2_diabetes":   {"brazil": 13000, "mexico": 12000, "colombia": 4000, "argentina": 3000, "chile": 1000},
        "respiratory":      {"brazil": 4000, "mexico": 3000, "colombia": 1500, "argentina": 1000, "chile": 400},
        "infectious":       {"brazil": 20000, "mexico": 10000, "colombia": 5000, "argentina": 2000, "chile": 1000},
    }

    # Find best matching indication category
    ind_lower = indication.lower()
    matched_category = next(
        (cat for cat in _prevalence_k if cat in ind_lower or ind_lower in cat),
        "oncology"
    )

    pool = {}
    for country in countries:
        pool[country] = {
            "estimated_eligible_k": _prevalence_k[matched_category].get(country, 100),
            "screening_rate_pct": 2.5,
            "estimated_screenable": int(_prevalence_k[matched_category].get(country, 100) * 1000 * 0.025),
        }

    total_screenable = sum(v["estimated_screenable"] for v in pool.values())
    feasibility = "HIGH" if total_screenable > target_n * 5 else "MODERATE" if total_screenable > target_n * 2 else "LOW"

    ctx.results["patient_pool"] = pool
    ctx.results["total_screenable"] = total_screenable
    ctx.results["enrollment_feasibility"] = feasibility
    return ctx


# ---------------------------------------------------------------------------
# Step 4: Regulatory Timeline
# ---------------------------------------------------------------------------
@step("trial_regulatory_timeline")
async def regulatory_timeline_step(ctx: PipelineContext) -> PipelineContext:
    """Build regulatory authorization timeline per LATAM country."""
    countries = ctx.results.get("target_countries", [])
    timeline = {}

    for country in countries:
        auth = _LATAM_TRIAL_AUTHORITIES.get(country.lower())
        if auth:
            timeline[country] = {
                "authority": auth["body"],
                "authorization_days": auth["timeline_days"],
                "submission_system": auth["system"],
                "parallel_ethics_review": True,
                "note": "Ethics + regulatory authorization can run in parallel",
            }

    max_days = max((v["authorization_days"] for v in timeline.values()), default=90)
    ctx.results["regulatory_timeline"] = timeline
    ctx.results["longest_auth_days"] = max_days
    return ctx


# ---------------------------------------------------------------------------
# Step 5: Risk Assessment
# ---------------------------------------------------------------------------
@step("trial_risk_assessment")
async def risk_assessment_step(ctx: PipelineContext) -> PipelineContext:
    """Assess trial risks per country and provide mitigations."""
    countries = ctx.results.get("target_countries", [])
    risks = []

    country_risks = {
        "brazil": [
            "CONEP review can extend 90→180 days for complex protocols",
            "Currency controls (BRL) complicate USD-denominated site payments",
            "Portuguese-language protocol translation required",
        ],
        "mexico": [
            "COFEPRIS authorization can extend to 180 days for oncology trials",
            "IMSS/ISSSTE site enrollment requires separate government approvals",
            "Tax withholding on investigator payments requires local fiscal setup",
        ],
        "colombia": [
            "Recent INVIMA reforms mean regulatory timelines are stabilizing",
            "Limited Phase III site infrastructure outside Bogotá/Medellín",
        ],
        "argentina": [
            "Economic instability (ARS devaluation) complicates multi-year budget planning",
            "ANMAT authorization requires prior approval before ethics submission",
            "Import restrictions may affect investigational drug supply",
        ],
        "chile": [
            "Smallest patient pool — may need 2+ countries for adequate enrollment",
            "ISP requires GMP certification for IMP importation",
        ],
    }

    for country in countries:
        for risk in country_risks.get(country.lower(), []):
            risks.append(f"[{country.upper()}] {risk}")

    ctx.results["risk_flags"] = risks
    ctx.metadata["risk_count"] = len(risks)
    ctx.final_content = self._build_summary(ctx)
    return ctx


# Patching the step — needs access to ctx after initialization
async def _risk_step_fixed(ctx: PipelineContext) -> PipelineContext:
    countries = ctx.results.get("target_countries", [])
    country_risks = {
        "brazil":    ["CONEP review can extend 90→180 days for complex protocols",
                      "Portuguese-language protocol translation required"],
        "mexico":    ["COFEPRIS authorization can extend to 180 days for oncology",
                      "IMSS/ISSSTE site enrollment requires separate government approvals"],
        "colombia":  ["Limited Phase III infrastructure outside major cities"],
        "argentina": ["Currency instability complicates multi-year budget planning",
                      "ANMAT authorization required before ethics submission"],
        "chile":     ["Smallest LATAM patient pool — plan multi-country enrollment"],
    }
    risks = []
    for country in countries:
        for risk in country_risks.get(country.lower(), []):
            risks.append(f"[{country.upper()}] {risk}")
    ctx.results["risk_flags"] = risks
    ctx.metadata["risk_count"] = len(risks)

    # Assemble summary
    pool = ctx.results.get("patient_pool", {})
    sites = ctx.results.get("recommended_sites", [])
    timeline = ctx.results.get("regulatory_timeline", {})
    feasibility = ctx.results.get("enrollment_feasibility", "UNKNOWN")

    summary = (
        f"\n\nCLINICAL TRIAL INTELLIGENCE SUMMARY\n{'='*50}\n"
        f"Indication: {ctx.results.get('indication')} | Phase {ctx.results.get('phase')}\n"
        f"Target N: {ctx.results.get('target_n')} | Enrollment Feasibility: {feasibility}\n"
        f"Sites Recommended: {len(sites)}\n"
        f"Total Screenable Patients: {ctx.results.get('total_screenable', 0):,}\n\n"
        f"REGULATORY AUTHORIZATION TIMELINES\n"
    )
    for country, info in timeline.items():
        summary += f"  {country.upper()}: {info['authority']} — {info['authorization_days']} days\n"

    summary += f"\nRISK FLAGS ({len(risks)} identified)\n"
    for risk in risks:
        summary += f"  ⚠ {risk}\n"

    ctx.final_content = ctx.results.get("protocol_analysis", "") + summary
    return ctx


# Replace the step function with the fixed version
risk_step = PipelineStep("trial_risk_assessment", _risk_step_fixed)

# ---------------------------------------------------------------------------
# Assembled Pipeline
# ---------------------------------------------------------------------------
clinical_trial_pipeline = Pipeline(
    steps=[
        protocol_analysis_step,
        site_recommendation_step,
        patient_pool_step,
        regulatory_timeline_step,
        risk_step,
    ],
    name="clinical_trial_latam",
)
