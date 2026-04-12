"""
Phase 3 API — Clinical Trial Intelligence + FDA 510(k) + Drug Discovery + International Regulatory.

Phase 3 capabilities:
  - /api/v1/phase3/trial/analyze         — LATAM clinical trial intelligence
  - /api/v1/phase3/trial/authorities     — LATAM regulatory authority reference
  - /api/v1/phase3/fda510k/prepare       — FDA 510(k) premarket notification package
  - /api/v1/phase3/fda510k/predicates    — Cleared AI predicate database
  - /api/v1/phase3/drug/targets          — AI drug target identification (OpenTargets + PubMed)
  - /api/v1/phase3/drug/repurposing      — Drug repurposing signal discovery
  - /api/v1/phase3/regulatory/expand     — International regulatory gap analysis (EMA/PMDA/MHRA/TGA)
  - /api/v1/phase3/regulatory/authorities — International authority registry
  - /api/v1/phase3/regulatory/ich        — ICH guideline applicability matrix
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from vinci_core.workflows.pipeline import PipelineContext
from vinci_core.workflows.clinical_trial_pipeline import clinical_trial_pipeline, _LATAM_TRIAL_AUTHORITIES
from vinci_core.workflows.fda_510k_pipeline import fda_510k_pipeline, _KNOWN_AI_PREDICATES
from vinci_core.drug_discovery.engine import drug_discovery_engine
from vinci_core.regulatory.international import (
    international_regulatory_engine,
    INTERNATIONAL_AUTHORITIES,
    ICH_GUIDELINES,
)

router = APIRouter(prefix="/phase3", tags=["Phase 3 — Drug Discovery + Trials + Regulatory"])


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


# ---------------------------------------------------------------------------
# Drug Discovery AI
# ---------------------------------------------------------------------------

class DrugTargetRequest(BaseModel):
    disease_area: str = Field(..., description="Disease area: type2_diabetes | chagas_disease | dengue | cardiovascular | oncology | leishmaniasis")
    countries: Optional[list[str]] = Field(None, description="LATAM countries for disease burden context")
    max_targets: int = Field(3, ge=1, le=5, description="Number of targets to return")
    use_opentargets: bool = Field(True, description="Enrich with Open Targets Platform live data")
    use_pubmed: bool = Field(True, description="Enrich with PubMed literature evidence")


class RepurposingRequest(BaseModel):
    disease_area: str = Field(..., description="Target disease area for repurposing")
    existing_drug: Optional[str] = Field(None, description="Specific drug to check for repurposing signals")
    countries: Optional[list[str]] = Field(None, description="LATAM markets for prioritization")


@router.post("/drug/targets")
async def identify_drug_targets(req: DrugTargetRequest):
    """
    AI Drug Target Identification.

    Combines:
    - Curated LATAM target database (gene → protein → disease)
    - Open Targets Platform (real-time: drug/target → disease associations)
    - PubMed literature mining (RAG-enriched evidence synthesis)
    - AI mechanistic analysis with LATAM disease burden context

    Returns ranked TargetHypothesis objects with druggability,
    development precedence, recommended modalities, and economics.
    """
    try:
        hypotheses = await drug_discovery_engine.identify_targets(
            disease_area=req.disease_area,
            countries=req.countries or [],
            max_targets=req.max_targets,
            use_opentargets=req.use_opentargets,
            use_pubmed=req.use_pubmed,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Drug discovery engine error: {e}")

    return {
        "disease_area": req.disease_area,
        "targets": [
            {
                "hypothesis_id": h.hypothesis_id,
                "gene_symbol": h.gene_symbol,
                "protein_name": h.protein_name,
                "druggability": h.druggability,
                "development_precedence": h.development_precedence,
                "recommended_modalities": h.recommended_modalities,
                "confidence_score": h.confidence_score,
                "development_economics": h.development_economics,
                "latam_relevance": h.latam_relevance,
                "evidence_sources": h.evidence_sources,
                "opentargets_hits": len(h.opentargets_associations),
                "pubmed_abstracts": len(h.pubmed_abstracts),
                "ai_analysis": h.ai_analysis,
                "generated_at": h.generated_at,
            }
            for h in hypotheses
        ],
        "total_targets": len(hypotheses),
        "note": "AI-generated hypotheses. Require experimental validation before clinical use.",
    }


@router.post("/drug/repurposing")
async def find_repurposing_candidates(req: RepurposingRequest):
    """
    Drug Repurposing Signal Discovery.

    Surfaces existing drugs with evidence for new LATAM disease indications.
    Uses Open Targets drug→disease associations and curated LATAM repurposing signals.

    High ROI: repurposed drugs skip Phase I safety studies (known safety profile),
    reducing development cost from $2B → $300M and timeline from 12 → 6 years.
    """
    try:
        candidates = await drug_discovery_engine.find_repurposing_candidates(
            disease_area=req.disease_area,
            existing_drug=req.existing_drug,
            countries=req.countries or [],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Repurposing engine error: {e}")

    return {
        "disease_area": req.disease_area,
        "candidates": [
            {
                "candidate_id": c.candidate_id,
                "drug_name": c.drug_name,
                "original_indication": c.original_indication,
                "proposed_indication": c.proposed_indication,
                "mechanism": c.mechanism,
                "repurposing": c.repurposing,
                "clinical_phase_original": c.clinical_phase_original,
                "evidence_score": c.evidence_score,
                "latam_markets": c.latam_markets,
                "ai_rationale": c.ai_rationale,
            }
            for c in candidates
        ],
        "total_candidates": len(candidates),
        "note": "Repurposing signals require clinical validation. Regulatory pathway varies by jurisdiction.",
    }


# ---------------------------------------------------------------------------
# International Regulatory Expansion
# ---------------------------------------------------------------------------

class InternationalExpansionRequest(BaseModel):
    product_type: str = Field(..., description="small_molecule | biologic | SaMD | combination")
    existing_approvals: list[str] = Field(
        default_factory=list,
        description="Already-approved authorities: anvisa | cofepris | invima | anmat | isp | fda | ema"
    )
    target_authorities: list[str] = Field(
        ...,
        description="Authorities to expand into: ema | pmda | mhra | tga | health_canada"
    )
    indication: str = Field(..., description="Therapeutic indication")
    is_samd: bool = Field(False, description="True if Software as a Medical Device")


@router.post("/regulatory/expand")
async def analyze_international_expansion(req: InternationalExpansionRequest):
    """
    International Regulatory Gap Analysis.

    Analyzes gaps between your existing LATAM approvals and target global markets.
    Covers EMA (EU), PMDA (Japan), MHRA (UK), TGA (Australia), Health Canada.

    Returns per-authority gap lists, required bridging studies, cost/timeline
    estimates, and parallel submission strategy (Access Consortium work-sharing).

    Strategic use: Phase 3 international expansion after LATAM revenue validates product.
    Enterprise contract value: $500K–$2M/year per pharma partner.
    """
    valid_authorities = set(INTERNATIONAL_AUTHORITIES.keys())
    invalid = [a for a in req.target_authorities if a.lower() not in valid_authorities]
    if invalid:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown authorities: {invalid}. Valid: {sorted(valid_authorities)}",
        )

    try:
        gaps = international_regulatory_engine.analyze_expansion(
            product_type=req.product_type,
            existing_approvals=req.existing_approvals,
            target_authorities=req.target_authorities,
            indication=req.indication,
            is_samd=req.is_samd,
        )
        strategy = international_regulatory_engine.get_parallel_submission_strategy(
            target_authorities=req.target_authorities,
            product_type=req.product_type,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"International regulatory engine error: {e}")

    return {
        "product_type": req.product_type,
        "indication": req.indication,
        "existing_approvals": req.existing_approvals,
        "gap_analyses": [
            {
                "authority": g.authority,
                "authority_name": g.authority_name,
                "region": g.region,
                "recommended_pathway": g.recommended_pathway,
                "gaps": g.gaps,
                "required_studies": g.required_studies,
                "estimated_bridging_cost_usd": g.estimated_bridging_cost_usd,
                "estimated_timeline_months": g.estimated_timeline_months,
                "parallel_submission_eligible": g.parallel_submission_eligible,
                "applicable_ich_guidelines": g.applicable_ich_guidelines,
                "expedited_programs": g.expedited_programs,
            }
            for g in gaps
        ],
        "parallel_submission_strategy": strategy,
    }


@router.get("/regulatory/authorities")
async def get_international_authorities(authority: Optional[str] = None):
    """
    International regulatory authority registry.
    Covers EMA, PMDA, MHRA, TGA, Health Canada with submission pathways and timelines.
    """
    if authority:
        auth = INTERNATIONAL_AUTHORITIES.get(authority.lower())
        if not auth:
            raise HTTPException(
                status_code=404,
                detail=f"Unknown authority '{authority}'. Valid: {sorted(INTERNATIONAL_AUTHORITIES.keys())}",
            )
        return {authority.lower(): auth}
    return INTERNATIONAL_AUTHORITIES


@router.get("/regulatory/ich")
async def get_ich_guidelines(authority: Optional[str] = None):
    """
    ICH guideline applicability matrix.
    Returns which ICH guidelines (E3, E6, E2B, M4, etc.) apply to each authority.
    """
    if authority:
        return {
            "authority": authority,
            "applicable_guidelines": international_regulatory_engine.get_ich_applicability(authority),
        }
    return {
        "guidelines": ICH_GUIDELINES,
        "note": "All major authorities (EMA, PMDA, MHRA, FDA, TGA, Health Canada) require ICH CTD format (M4)",
    }
