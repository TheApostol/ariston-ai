"""
Individual Agent REST API — exposes DigitalTwin, IoMT, Regulatory Copilot,
PGx, Patient, PharmacovigilanceNarrative, and SiteSelection agents as
standalone FastAPI endpoints.

Endpoints:
  POST /agents/twin/simulate              — digital twin treatment simulation
  POST /agents/iomt/adherence             — IoMT 30-day adherence forecast
  POST /agents/pgx/cross-reference        — PGx gene/drug interaction check
  POST /agents/regulatory/report          — GxP regulatory copilot report
  GET  /agents/patient/{id}/history       — patient longitudinal history
  POST /agents/patient/{id}/record        — add patient record
  POST /agents/pv/narrative               — CIOMS / MedWatch narrative generation
  POST /agents/pv/narrative/batch         — batch narrative generation
  POST /agents/sites/recommend            — LatAm site selection recommendations
  POST /agents/sites/feasibility          — multi-country feasibility summary
  GET  /agents/health                     — health check
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

from vinci_core.agent.twin_agent import digital_twin_agent
from vinci_core.agent.iomt_agent import iomt_agent
from vinci_core.agent.genomics_agent import pharmacogenomics_agent
from vinci_core.agent.regulatory_agent import regulatory_copilot
from vinci_core.agent.patient_agent import patient_agent
from vinci_core.agent.pv_narrative_agent import pv_narrative_agent
from vinci_core.agent.site_selection_agent import site_selection_agent

router = APIRouter(prefix="/agents", tags=["Individual Agents"])


# ── Request / Response models ─────────────────────────────────────────────────

class TwinSimulateRequest(BaseModel):
    drug: str
    patient_history: str
    genetics: Optional[List[str]] = None


class IoMTAdherenceRequest(BaseModel):
    patient_history: str
    telemetry: Optional[Dict[str, Any]] = None


class PGxRequest(BaseModel):
    drug_name: str
    patient_id: Optional[str] = None


class RegulatoryReportRequest(BaseModel):
    prompt: str
    result: str
    job_id: Optional[str] = None
    audit_logs: Optional[List[Dict[str, Any]]] = None


class PatientRecordRequest(BaseModel):
    event_type: str
    details: str
    date: Optional[str] = None


class PVNarrativeRequest(BaseModel):
    """Adverse event data for CIOMS / MedWatch narrative generation."""
    case_id: str
    drug_name: str
    dose: str
    indication: str
    ae_term: str                        # MedDRA preferred term
    onset_date: str
    outcome: str                        # "recovered" | "recovering" | "not recovered" | "fatal" | "unknown"
    severity: str = "non-serious"       # see PharmacovigilanceNarrativeAgent.SEVERITY_MAP
    patient_age: Optional[int] = None
    patient_sex: Optional[str] = None
    medical_history: Optional[str] = None
    reporter_type: str = "physician"
    narrative: Optional[str] = None
    format: str = "both"                # "cioms" | "medwatch" | "both"


class PVBatchRequest(BaseModel):
    events: List[Dict[str, Any]]
    format: str = "cioms"               # "cioms" | "medwatch" | "both"


class SiteRecommendRequest(BaseModel):
    therapeutic_area: str
    agency: Optional[str] = None        # ANVISA | COFEPRIS | INVIMA | ANMAT
    country: Optional[str] = None
    top_n: int = 5
    min_score: float = 40.0


class SiteFeasibilityRequest(BaseModel):
    therapeutic_area: str
    agencies: Optional[List[str]] = None   # default: all 4 LatAm agencies


# ── Digital Twin ──────────────────────────────────────────────────────────────

@router.post("/twin/simulate")
async def twin_simulate(request: TwinSimulateRequest):
    """
    Run an in-silico Digital Twin treatment simulation.

    Scores predicted efficacy, toxicity risk, and organ impact
    based on patient history and known genetic variants.

    Simulation engine: Ariston-Twin-V2 (Monte Carlo Heuristic, 5000 iterations)

    Example:
      POST /api/v1/agents/twin/simulate
      {
        "drug": "clopidogrel",
        "patient_history": "68-year-old with renal failure, CKD stage 3, chest pain",
        "genetics": ["CYP2C19 Poor Metabolizer"]
      }
    """
    result = digital_twin_agent.simulate_treatment(
        history=request.patient_history,
        drug=request.drug,
        genetics=request.genetics or [],
    )
    return {
        "drug": request.drug,
        "simulation": result,
    }


# ── IoMT Adherence ────────────────────────────────────────────────────────────

@router.post("/iomt/adherence")
async def iomt_adherence(request: IoMTAdherenceRequest):
    """
    Forecast patient medication adherence over 30 days.

    Uses IoMT device telemetry (pillbox opens, heart rate, steps)
    combined with patient history to predict adherence risk.

    Example:
      POST /api/v1/agents/iomt/adherence
      {
        "patient_history": "72-year-old with early dementia",
        "telemetry": { "pillbox_opens_7d": 3, "avg_heart_rate": 78, "steps_daily": 2500 }
      }
    """
    result = iomt_agent.forecast_adherence(
        history=request.patient_history,
        telemetry=request.telemetry,
    )
    return result


# ── Pharmacogenomics ──────────────────────────────────────────────────────────

@router.post("/pgx/cross-reference")
async def pgx_cross_reference(request: PGxRequest):
    """
    Cross-reference a drug against the PGx gene interaction database.

    Checks CYP2C19, CYP2D6, HLA-B*5701, HLA-B*1502, TPMT, VKORC1, DPYD, UGT1A1
    against live ClinVar data.

    Example:
      POST /api/v1/agents/pgx/cross-reference
      { "drug_name": "clopidogrel" }
    """
    return await pharmacogenomics_agent.cross_reference(
        drug_name=request.drug_name,
        patient_id=request.patient_id,
    )


# ── Regulatory Copilot ────────────────────────────────────────────────────────

@router.post("/regulatory/report")
async def regulatory_report(request: RegulatoryReportRequest):
    """
    Generate a GxP-compliant Ariston Clinical Report (ACR).

    Suitable for IRB submission packages. Includes:
    - Executive summary
    - Decision proof (grounding)
    - Regulatory safety checks (HIPAA/GDPR, GxP, PGx, IoMT)
    - Audit summary
    - Compliance signature with SHA-256 integrity hash

    Example:
      POST /api/v1/agents/regulatory/report
      {
        "prompt": "Phase 2 clopidogrel safety review",
        "result": "Safety analysis text..."
      }
    """
    import uuid
    job_id = request.job_id or str(uuid.uuid4())
    audit_logs = request.audit_logs or []

    gxp_report = regulatory_copilot.generate_report(
        job_id=job_id,
        prompt=request.prompt,
        result=request.result,
        audit_logs=audit_logs,
    )
    return {
        "job_id": job_id,
        "gxp_report": gxp_report,
        "compliance": "GxP / FDA 21 CFR Part 11 / EU Annex 11",
    }


# ── Patient History ───────────────────────────────────────────────────────────

@router.get("/patient/{patient_id}/history")
async def get_patient_history(patient_id: str):
    """Get the longitudinal clinical history for a patient."""
    history = patient_agent.get_full_history(patient_id)
    return {
        "patient_id": patient_id,
        "history": history,
        "has_records": bool(history),
    }


@router.post("/patient/{patient_id}/record")
async def add_patient_record(patient_id: str, request: PatientRecordRequest):
    """Add a clinical event to a patient's longitudinal history."""
    patient_agent.add_record(
        patient_id=patient_id,
        event_type=request.event_type,
        details=request.details,
        date=request.date,
    )
    return {"status": "ok", "patient_id": patient_id}


@router.get("/health")
async def health():
    return {
        "status": "ok",
        "available_agents": [
            "digital_twin", "iomt", "pharmacogenomics",
            "regulatory_copilot", "patient_history",
            "pv_narrative", "site_selection",
        ],
    }


# ── Pharmacovigilance Narrative ───────────────────────────────────────────────

@router.post("/pv/narrative")
async def pv_narrative(request: PVNarrativeRequest):
    """
    Generate a CIOMS-I and/or MedWatch adverse event narrative.

    Format options:
      - "cioms"    → CIOMS-I ICSR format
      - "medwatch" → FDA 3500A equivalent
      - "both"     → both formats in one response

    Example:
      POST /api/v1/agents/pv/narrative
      {
        "case_id": "AE-2024-001",
        "drug_name": "semaglutide",
        "dose": "1 mg weekly subcutaneous",
        "indication": "Type 2 Diabetes",
        "ae_term": "Pancreatitis",
        "onset_date": "2024-03-15",
        "outcome": "recovering",
        "severity": "hospitalization",
        "patient_age": 62,
        "patient_sex": "male",
        "format": "both"
      }
    """
    event = request.model_dump(exclude={"format"})
    fmt = request.format

    if fmt == "medwatch":
        return {"case_id": request.case_id, "medwatch": pv_narrative_agent.generate_medwatch(event)}
    elif fmt == "both":
        return {"case_id": request.case_id, **pv_narrative_agent.generate_both(event)}
    else:
        return {"case_id": request.case_id, "cioms": pv_narrative_agent.generate_cioms(event)}


@router.post("/pv/narrative/batch")
async def pv_narrative_batch(request: PVBatchRequest):
    """
    Generate narratives for a batch of adverse events.

    Accepts a list of event dicts (same schema as /pv/narrative).
    Returns a list of narratives keyed by case_id.

    Example:
      POST /api/v1/agents/pv/narrative/batch
      { "events": [...], "format": "cioms" }
    """
    results = pv_narrative_agent.batch_generate(request.events, format=request.format)
    return {
        "total": len(results),
        "format": request.format,
        "narratives": results,
    }


# ── Site Selection ────────────────────────────────────────────────────────────

@router.post("/sites/recommend")
async def sites_recommend(request: SiteRecommendRequest):
    """
    Recommend the best LatAm clinical trial sites for a given therapeutic area.

    Scoring (100-point scale):
      Therapeutic area match (30) + Capacity (25) + Investigator tier (20)
      + Approval speed (15) + Patient pool (10)

    Example:
      POST /api/v1/agents/sites/recommend
      { "therapeutic_area": "oncology", "agency": "ANVISA", "top_n": 3 }
    """
    return site_selection_agent.recommend_sites(
        therapeutic_area=request.therapeutic_area,
        agency=request.agency,
        country=request.country,
        top_n=request.top_n,
        min_score=request.min_score,
    )


@router.post("/sites/feasibility")
async def sites_feasibility(request: SiteFeasibilityRequest):
    """
    Generate a multi-country LatAm feasibility summary for a therapeutic area.

    Returns per-agency top site and an overall readiness tier (HIGH / MODERATE / LOW).

    Example:
      POST /api/v1/agents/sites/feasibility
      { "therapeutic_area": "oncology" }
    """
    return site_selection_agent.feasibility_summary(
        therapeutic_area=request.therapeutic_area,
        agencies=request.agencies,
    )
