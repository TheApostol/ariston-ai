"""
Phase 3 API — Clinical Trial Intelligence + FDA 510(k) Preparation / Ariston AI.

Phase 3 capabilities:
  - /api/v1/phase3/trial/analyze    — LATAM clinical trial intelligence (protocol → sites → patients → timeline → risks)
  - /api/v1/phase3/fda510k/prepare  — FDA 510(k) premarket notification package generation
  - /api/v1/phase3/fda510k/predicates — Known cleared AI predicates reference list

Phase 3 roadmap milestones:
  - Biomarker discovery AI (leverages RWE from Phase 2)
  - Clinical decision support (510(k) pathway)
  - International expansion beyond LATAM
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from vinci_core.workflows.pipeline import PipelineContext
from vinci_core.workflows.clinical_trial_pipeline import clinical_trial_pipeline, _LATAM_TRIAL_AUTHORITIES
from vinci_core.workflows.fda_510k_pipeline import fda_510k_pipeline, _KNOWN_AI_PREDICATES

router = APIRouter(prefix="/phase3", tags=["Phase 3 — Clinical Trials + FDA 510(k)"])


# ---------------------------------------------------------------------------
# Clinical Trial Intelligence
# ---------------------------------------------------------------------------

class ClinicalTrialRequest(BaseModel):
    prompt: str = Field(default="Analyze clinical trial feasibility", description="Analysis request or question")
    protocol_data: Optional[dict] = Field(
        None,
        description=(
            "Protocol parameters: phase (II/III), indication, arms, "
            "target_enrollment, duration_months, countries (list of LATAM countries)"
        ),
        examples=[{
            "phase": "II/III",
            "indication": "oncology",
            "arms": 2,
            "target_enrollment": 300,
            "duration_months": 24,
            "countries": ["brazil", "mexico", "colombia"],
        }],
    )


@router.post("/trial/analyze")
async def analyze_clinical_trial(req: ClinicalTrialRequest):
    """
    LATAM Clinical Trial Intelligence.

    Runs a 5-step pipeline:
    1. Protocol feasibility analysis (AI-powered)
    2. Site recommendation and scoring
    3. Patient pool sizing per country
    4. Regulatory authorization timeline (ANVISA/COFEPRIS/INVIMA/ANMAT/ISP)
    5. Country-specific risk flags and mitigations

    Target buyers: Sponsors and CROs running Phase II–III trials in LATAM.
    Value: $100K–$500K per customer per trial.
    """
    ctx = PipelineContext(
        prompt=req.prompt,
        layer="pharma",
        metadata={"protocol_data": req.protocol_data or {}},
    )
    try:
        result = await clinical_trial_pipeline.run(ctx)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clinical trial pipeline error: {e}")

    return {
        "indication": result.results.get("indication"),
        "phase": result.results.get("phase"),
        "target_n": result.results.get("target_n"),
        "enrollment_feasibility": result.results.get("enrollment_feasibility"),
        "total_screenable_patients": result.results.get("total_screenable"),
        "recommended_sites": result.results.get("recommended_sites", []),
        "site_count": result.results.get("site_count", 0),
        "patient_pool": result.results.get("patient_pool", {}),
        "regulatory_timeline": result.results.get("regulatory_timeline", {}),
        "longest_authorization_days": result.results.get("longest_auth_days"),
        "risk_flags": result.results.get("risk_flags", []),
        "risk_count": result.metadata.get("risk_count", 0),
        "protocol_analysis": result.results.get("protocol_analysis"),
        "full_summary": result.final_content,
        "metadata": result.metadata,
    }


@router.get("/trial/authorities")
async def get_latam_trial_authorities(country: Optional[str] = None):
    """
    Return LATAM clinical trial regulatory authorities and authorization timelines.
    Covers Brazil (ANVISA+CONEP), Mexico (COFEPRIS+CONBIOÉTICA), Colombia (INVIMA),
    Argentina (ANMAT), Chile (ISP).
    """
    if country:
        auth = _LATAM_TRIAL_AUTHORITIES.get(country.lower())
        if not auth:
            raise HTTPException(
                status_code=404,
                detail=f"Country '{country}' not found. Available: {list(_LATAM_TRIAL_AUTHORITIES.keys())}",
            )
        return {country.lower(): auth}
    return _LATAM_TRIAL_AUTHORITIES


# ---------------------------------------------------------------------------
# FDA 510(k) Preparation
# ---------------------------------------------------------------------------

class FDA510kRequest(BaseModel):
    prompt: str = Field(default="Prepare FDA 510(k) submission package", description="510(k) preparation request")
    device_data: Optional[dict] = Field(
        None,
        description=(
            "Device metadata: device_name, indication, device_type, "
            "predicate_k_number (optional), target_population"
        ),
        examples=[{
            "device_name": "AI Clinical Decision Support System",
            "indication": "chest x-ray triage for pneumonia detection",
            "device_type": "radiology AI",
            "target_population": "adult inpatients",
        }],
    )


@router.post("/fda510k/prepare")
async def prepare_510k_submission(req: FDA510kRequest):
    """
    FDA 510(k) Premarket Notification Package Generator.

    Generates a complete 510(k) shell document including:
    1. Predicate device identification (from cleared AI algorithm database)
    2. Intended Use and Indications for Use statements
    3. Substantial Equivalence argument (SE argument per 513(i) FD&C Act)
    4. Predetermined Change Control Plan (PCCP) for adaptive AI
    5. Assembled submission package shell

    Regulatory strategy:
    - Phase 1: Non-device CDS (Cures Act CDS exemption) — no 510(k) needed
    - Phase 3: 510(k) for diagnostic aid — your cleared device becomes the predicate for competitors

    Estimated timeline: 142 days median FDA review.
    Estimated cost: $150K–$500K submission preparation.
    """
    ctx = PipelineContext(
        prompt=req.prompt,
        layer="pharma",
        metadata={"device_data": req.device_data or {}},
    )
    try:
        result = await fda_510k_pipeline.run(ctx)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"FDA 510(k) pipeline error: {e}")

    return {
        "predicate_candidates": result.results.get("predicate_candidates", []),
        "indication": result.results.get("indication"),
        "intended_use_draft": result.results.get("intended_use_draft"),
        "se_argument": result.results.get("se_argument"),
        "pccp_draft": result.results.get("pccp_draft"),
        "submission_package": result.final_content,
        "regulatory_notes": {
            "pathway": "510(k) — Substantial Equivalence",
            "median_review_days": 142,
            "estimated_cost_usd": "150000–500000",
            "cds_exemption_applicable": "Assess per Cures Act 4-prong criteria in intended use draft",
            "pccp_required": True,
        },
        "metadata": result.metadata,
    }


@router.get("/fda510k/predicates")
async def get_known_ai_predicates(indication: Optional[str] = None):
    """
    Return known cleared AI/ML medical device predicates from the FDA 510(k) database.
    Use these to identify predicate devices for your 510(k) submission.
    Over 1,250 AI algorithms have been cleared as of 2024.
    """
    if indication:
        ind_lower = indication.lower()
        keywords = ind_lower.split()[:4]
        filtered = [
            p for p in _KNOWN_AI_PREDICATES
            if any(kw in p["indication"] for kw in keywords)
            or any(kw in p["device"].lower() for kw in keywords)
        ]
        return {
            "query": indication,
            "matches": filtered,
            "total_matches": len(filtered),
            "note": "Production: query FDA 510(k) database API at accessdata.fda.gov for full predicate search",
        }
    return {
        "predicates": _KNOWN_AI_PREDICATES,
        "total": len(_KNOWN_AI_PREDICATES),
        "note": "Curated subset. Full database: 1,250+ cleared AI algorithms at accessdata.fda.gov",
    }
