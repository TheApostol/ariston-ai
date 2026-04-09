"""
Phase 2 API — Real-World Evidence + Pharmacovigilance + CSR / Ariston AI.

Consolidates Phase 2 capabilities:
  - /api/v1/phase2/pv/report    — Pharmacovigilance narrative (CIOMS-I/MedWatch)
  - /api/v1/phase2/csr/generate — Clinical Study Report draft
  - /api/v1/phase2/rwe/*        — Real-World Evidence insights (see rwe/router.py)
  - /api/v1/phase2/biomarker/*  — Biomarker discovery hypotheses (Phase 3 preview)
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from vinci_core.workflows.pipeline import PipelineContext
from vinci_core.workflows.pharmacovigilance_pipeline import pharmacovigilance_pipeline
from vinci_core.workflows.csr_pipeline import csr_pipeline
from vinci_core.biomarker.discovery import biomarker_engine, BIOMARKER_TYPES, LATAM_DISEASE_PRIORITIES

router = APIRouter(prefix="/phase2", tags=["Phase 2 — RWE + PhV + CSR"])


# ---------------------------------------------------------------------------
# Pharmacovigilance
# ---------------------------------------------------------------------------

class PVReportRequest(BaseModel):
    prompt: str = Field(..., description="Adverse event description or structured AE data request")
    ae_data: Optional[dict] = Field(None, description="Structured AE dict: drug_name, ae_term, seriousness, outcome, countries")


@router.post("/pv/report")
async def generate_pv_report(req: PVReportRequest):
    """
    Generate CIOMS-I and MedWatch pharmacovigilance narratives.
    Includes LATAM agency routing plan (VIGIMED, FARMACOVIGILANCIA, SNVS, etc.).
    """
    ctx = PipelineContext(
        prompt=req.prompt,
        layer="pharma",
        metadata={"ae_data": req.ae_data or {}},
    )
    try:
        result = await pharmacovigilance_pipeline.run(ctx)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PV pipeline error: {e}")

    return {
        "case_id": result.results.get("ae_case_id"),
        "cioms_narrative": result.results.get("cioms_narrative"),
        "medwatch_narrative": result.results.get("medwatch_narrative"),
        "meddra_coding": {
            "soc": result.results.get("meddra_soc"),
            "code": result.results.get("meddra_code"),
            "pt": result.results.get("meddra_pt"),
        },
        "causality_who_umc": result.results.get("causality_who_umc"),
        "latam_reporting_plan": result.results.get("latam_reporting_plan"),
        "gxp_signed": result.metadata.get("gxp_signed", False),
        "gxp_hash": result.results.get("gxp_hash"),
        "full_report": result.final_content,
        "metadata": result.metadata,
    }


# ---------------------------------------------------------------------------
# Clinical Study Report
# ---------------------------------------------------------------------------

class CSRRequest(BaseModel):
    prompt: str = Field(default="Generate clinical study report", description="CSR generation request")
    study_data: Optional[dict] = Field(
        None,
        description=(
            "Study metadata: title, number, drug_name, indication, phase, "
            "sponsor, study_period, subject_count, primary_endpoint, "
            "countries, ae_summary"
        ),
    )


@router.post("/csr/generate")
async def generate_csr(req: CSRRequest):
    """
    Generate an ICH E3-compliant Clinical Study Report draft.
    Includes LATAM regulatory adaptation notes per country.
    Time savings: 180 → ~80 hours (55% reduction per McKinsey/Merck benchmark).
    """
    ctx = PipelineContext(
        prompt=req.prompt,
        layer="pharma",
        metadata={"study_data": req.study_data or {}},
    )
    try:
        result = await csr_pipeline.run(ctx)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CSR pipeline error: {e}")

    return {
        "study_number": result.results.get("study_number"),
        "study_title": result.results.get("study_title"),
        "drug_name": result.results.get("drug_name"),
        "latam_countries": result.results.get("latam_countries"),
        "csr_draft": result.final_content,
        "latam_notes": result.results.get("latam_adaptation_notes"),
        "metadata": result.metadata,
    }


# ---------------------------------------------------------------------------
# Biomarker Discovery (Phase 3 preview)
# ---------------------------------------------------------------------------

class BiomarkerRequest(BaseModel):
    disease_area: str
    biomarker_type: str = "predictive"
    countries: Optional[list[str]] = None
    use_rwe: bool = False


@router.post("/biomarker/hypotheses")
async def generate_biomarker_hypotheses(req: BiomarkerRequest):
    """
    Generate AI-assisted biomarker hypotheses for a disease area.
    Phase 3 capability — uses accumulated RWE data + PubMed RAG.
    """
    if req.biomarker_type not in BIOMARKER_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"biomarker_type must be one of: {BIOMARKER_TYPES}",
        )
    try:
        hypotheses = await biomarker_engine.generate_hypotheses(
            disease_area=req.disease_area,
            biomarker_type=req.biomarker_type,
            countries=req.countries or [],
            use_rwe=req.use_rwe,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Biomarker engine error: {e}")

    return {
        "hypotheses": [
            {
                "hypothesis_id": h.hypothesis_id,
                "biomarker_name": h.biomarker_name,
                "biomarker_type": h.biomarker_type,
                "disease_area": h.disease_area,
                "evidence_level": h.evidence_level,
                "confidence_score": h.confidence_score,
                "latam_relevance": h.latam_relevance,
                "ai_analysis": h.metadata.get("ai_analysis", ""),
            }
            for h in hypotheses
        ],
        "note": "These are AI-generated hypotheses requiring laboratory validation before clinical use.",
    }


@router.get("/biomarker/disease-priorities")
async def get_latam_disease_priorities(country: Optional[str] = None):
    """Return LATAM disease priority areas for biomarker research targeting."""
    return biomarker_engine.get_latam_disease_priorities(country=country)
