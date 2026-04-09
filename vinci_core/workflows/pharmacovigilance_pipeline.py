"""
Pharmacovigilance Pipeline — Phase 2 / Ariston AI LATAM.

Orchestrates adverse event detection, CIOMS narrative generation,
and regulatory submission for LATAM PhV obligations:
  - ANVISA: VIGIMED system (Brazil)
  - COFEPRIS: FAERS-equivalent PhV database (Mexico)
  - INVIMA: FARMACOVIGILANCIA portal (Colombia)
  - ANMAT: SNVS (Sistema Nacional de Vigilancia de la Salud, Argentina)
  - ISP: CENABAST PhV system (Chile)

Pipeline steps:
  1. AE Intake       — parse structured/unstructured adverse event reports
  2. MedDRA Coding   — assign preferred terms + SOC via ontology
  3. Causality       — WHO-UMC causality assessment (certain/probable/possible/unlikely)
  4. Narrative Gen   — CIOMS-I + MedWatch format via PharmacovigilanceNarrativeAgent
  5. LATAM Routing   — determine which LATAM agencies require submission + deadlines
  6. GxP Signing     — attach regulatory integrity hash via audit ledger

Timeframes (per ICH E2B and LATAM local requirements):
  - Serious unexpected: 15 calendar days (SUSAR)
  - Serious expected: 15 days (most LATAM agencies)
  - Non-serious: periodic (30–90 days depending on agency)
"""

from __future__ import annotations

from typing import Any, Optional
from vinci_core.workflows.pipeline import Pipeline, PipelineContext, PipelineStep, step
from vinci_core.agent.pv_narrative_agent import PharmacovigilanceNarrativeAgent

_pv_agent = PharmacovigilanceNarrativeAgent()

# Reporting windows by seriousness (calendar days)
_LATAM_REPORTING_WINDOWS = {
    "serious_unexpected": {
        "brazil":    15,   # ANVISA RDC 204/2017 — SUSAR 15 days
        "mexico":    15,   # COFEPRIS NOM-220-SSA1-2012
        "colombia":  15,   # INVIMA Resolución 2004009455
        "argentina": 15,   # ANMAT Disposición 5730/2010
        "chile":     15,   # ISP DS 3/2010
    },
    "serious_expected": {
        "brazil":    15,
        "mexico":    15,
        "colombia":  30,
        "argentina": 15,
        "chile":     30,
    },
    "non_serious": {
        "brazil":    90,
        "mexico":    90,
        "colombia":  90,
        "argentina": 90,
        "chile":     90,
    },
}

_LATAM_PV_PORTALS = {
    "brazil":    "VIGIMED (ANVISA) — vigimed.anvisa.gov.br",
    "mexico":    "COFEPRIS PhV Database — farmacovigilancia.cofepris.gob.mx",
    "colombia":  "FARMACOVIGILANCIA (INVIMA) — farmacovigilancia.invima.gov.co",
    "argentina": "SNVS (ANMAT) — snvs2.msal.gov.ar",
    "chile":     "ISP PhV System — ispch.cl/farmacovigilancia",
}


# ---------------------------------------------------------------------------
# Step 1: AE Intake
# ---------------------------------------------------------------------------
@step("pv_ae_intake")
async def ae_intake_step(ctx: PipelineContext) -> PipelineContext:
    """Parse adverse event data from prompt or structured context."""
    ae_data = ctx.metadata.get("ae_data") or {}

    # Extract key AE fields from context or prompt
    ctx.results["ae_case_id"]  = ae_data.get("case_id", f"AE-{ctx.metadata.get('request_id', 'UNKNOWN')[:8]}")
    ctx.results["drug_name"]   = ae_data.get("drug_name") or _extract_drug(ctx.prompt)
    ctx.results["ae_term"]     = ae_data.get("ae_term", "adverse event (unspecified)")
    ctx.results["seriousness"] = ae_data.get("seriousness", "serious_unexpected")
    ctx.results["outcome"]     = ae_data.get("outcome", "unknown")
    ctx.results["countries"]   = ae_data.get("countries") or _detect_countries(ctx.prompt)

    return ctx


# ---------------------------------------------------------------------------
# Step 2: MedDRA Coding
# ---------------------------------------------------------------------------
@step("pv_meddra_coding")
async def meddra_coding_step(ctx: PipelineContext) -> PipelineContext:
    """Assign MedDRA preferred term and SOC (stub — production uses MedDRA API)."""
    ae_term = ctx.results.get("ae_term", "")

    # Simplified MedDRA SOC mapping (production: integrate MedDRA browser API)
    _soc_map = {
        "hepatotoxicity":      ("Hepatobiliary disorders", "10019837"),
        "anaphylaxis":         ("Immune system disorders", "10021428"),
        "cardiac arrest":      ("Cardiac disorders", "10007515"),
        "rash":                ("Skin and subcutaneous tissue disorders", "10040785"),
        "nausea":              ("Gastrointestinal disorders", "10028813"),
        "neutropenia":         ("Blood and lymphatic system disorders", "10029359"),
        "adverse event (unspecified)": ("General disorders", "10018065"),
    }

    ae_lower = ae_term.lower()
    matched = None
    for key, (soc, code) in _soc_map.items():
        if key in ae_lower:
            matched = (soc, code)
            break

    ctx.results["meddra_soc"]  = matched[0] if matched else "General disorders and administration site conditions"
    ctx.results["meddra_code"] = matched[1] if matched else "10018065"
    ctx.results["meddra_pt"]   = ae_term

    return ctx


# ---------------------------------------------------------------------------
# Step 3: WHO-UMC Causality Assessment
# ---------------------------------------------------------------------------
@step("pv_causality")
async def causality_step(ctx: PipelineContext) -> PipelineContext:
    """Assign WHO-UMC causality category (simplified rule-based)."""
    ae_data = ctx.metadata.get("ae_data") or {}
    rechallenge = ae_data.get("rechallenge", False)
    dechallenge = ae_data.get("dechallenge", True)
    alternative_explanation = ae_data.get("alternative_explanation", False)

    if rechallenge and dechallenge and not alternative_explanation:
        causality = "certain"
    elif dechallenge and not alternative_explanation:
        causality = "probable"
    elif not alternative_explanation:
        causality = "possible"
    else:
        causality = "unlikely"

    ctx.results["causality_who_umc"] = causality
    return ctx


# ---------------------------------------------------------------------------
# Step 4: Narrative Generation (CIOMS-I + MedWatch)
# ---------------------------------------------------------------------------
@step("pv_narrative_generation")
async def narrative_generation_step(ctx: PipelineContext) -> PipelineContext:
    """Generate CIOMS-I and MedWatch narratives using PharmacovigilanceNarrativeAgent."""
    ae_data = ctx.metadata.get("ae_data") or {}

    event_payload = {
        "case_id":      ctx.results.get("ae_case_id"),
        "drug_name":    ctx.results.get("drug_name", "Unknown drug"),
        "dose":         ae_data.get("dose", "Not reported"),
        "indication":   ae_data.get("indication", "Not reported"),
        "ae_term":      ctx.results.get("ae_term"),
        "onset_date":   ae_data.get("onset_date", "Not reported"),
        "outcome":      ctx.results.get("outcome"),
        "patient_age":  ae_data.get("patient_age"),
        "patient_sex":  ae_data.get("patient_sex"),
        "meddra_soc":   ctx.results.get("meddra_soc"),
        "causality":    ctx.results.get("causality_who_umc"),
        "seriousness":  ctx.results.get("seriousness"),
    }

    cioms_narrative    = _pv_agent.generate_cioms(event_payload)
    medwatch_narrative = _pv_agent.generate_medwatch(event_payload)

    ctx.results["cioms_narrative"]    = cioms_narrative
    ctx.results["medwatch_narrative"] = medwatch_narrative
    ctx.final_content = cioms_narrative

    return ctx


# ---------------------------------------------------------------------------
# Step 5: LATAM Regulatory Routing
# ---------------------------------------------------------------------------
@step("pv_latam_routing")
async def latam_routing_step(ctx: PipelineContext) -> PipelineContext:
    """Determine reporting obligations per LATAM country and deadline."""
    countries     = ctx.results.get("countries", list(_LATAM_PV_PORTALS.keys()))
    seriousness   = ctx.results.get("seriousness", "serious_unexpected")
    reporting_plan = {}

    for country in countries:
        days = _LATAM_REPORTING_WINDOWS.get(seriousness, {}).get(country, 15)
        portal = _LATAM_PV_PORTALS.get(country, "national authority portal")
        reporting_plan[country] = {
            "portal": portal,
            "deadline_days": days,
            "format": "CIOMS-I (E2B R3 XML preferred)",
            "language": "Portuguese" if country == "brazil" else "Spanish",
        }

    ctx.results["latam_reporting_plan"] = reporting_plan
    ctx.metadata["agencies_to_notify"]  = list(reporting_plan.keys())

    routing_summary = "\n\n---\nLATAM PHARMACOVIGILANCE REPORTING PLAN\n"
    for country, plan in reporting_plan.items():
        routing_summary += (
            f"\n[{country.upper()}] {plan['portal']}\n"
            f"  Deadline: {plan['deadline_days']} calendar days\n"
            f"  Language: {plan['language']}\n"
        )

    ctx.final_content += routing_summary
    return ctx


# ---------------------------------------------------------------------------
# Step 6: GxP Signing
# ---------------------------------------------------------------------------
@step("pv_gxp_sign")
async def gxp_sign_step(ctx: PipelineContext) -> PipelineContext:
    """Attach GxP integrity hash and compliance metadata."""
    import hashlib, time
    content_hash = hashlib.sha256(ctx.final_content.encode()).hexdigest()

    ctx.results["gxp_hash"]      = content_hash
    ctx.results["gxp_timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    ctx.metadata["gxp_signed"]   = True

    ctx.final_content += (
        f"\n\n---\nGxP INTEGRITY SIGNATURE\n"
        f"Hash (SHA-256): {content_hash}\n"
        f"Signed at: {ctx.results['gxp_timestamp']}\n"
        f"Status: READY FOR REGULATORY SUBMISSION\n"
    )
    return ctx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _extract_drug(prompt: str) -> str:
    import re
    patterns = [r"drug[:\s]+([A-Za-z\-]+)", r"for ([A-Za-z\-]+) treatment", r"patient took ([A-Za-z\-]+)"]
    for pat in patterns:
        m = re.search(pat, prompt, re.IGNORECASE)
        if m:
            return m.group(1)
    return "Unknown drug"


def _detect_countries(prompt: str) -> list[str]:
    lower = prompt.lower()
    mapping = {"brazil": ["brazil", "brasil", "anvisa"], "mexico": ["mexico", "cofepris"],
               "colombia": ["colombia", "invima"], "argentina": ["argentina", "anmat"], "chile": ["chile", "isp"]}
    return [c for c, kws in mapping.items() if any(kw in lower for kw in kws)] or list(_LATAM_PV_PORTALS.keys())


# ---------------------------------------------------------------------------
# Assembled Pipeline
# ---------------------------------------------------------------------------
pharmacovigilance_pipeline = Pipeline(
    steps=[
        ae_intake_step,
        meddra_coding_step,
        causality_step,
        narrative_generation_step,
        latam_routing_step,
        gxp_sign_step,
    ],
    name="pharmacovigilance_latam",
)
