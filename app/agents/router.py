"""
Individual Agent REST API — exposes DigitalTwin, IoMT, Regulatory Copilot,
PGx, and Patient agents as standalone FastAPI endpoints.

Endpoints:
  POST /agents/twin/simulate        — digital twin treatment simulation
  POST /agents/iomt/adherence       — IoMT 30-day adherence forecast
  POST /agents/pgx/cross-reference  — PGx gene/drug interaction check
  POST /agents/regulatory/report    — GxP regulatory copilot report
  GET  /agents/patient/{id}/history — patient longitudinal history
  POST /agents/patient/{id}/record  — add patient record
  GET  /agents/health               — health check
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

from vinci_core.agent.twin_agent import digital_twin_agent
from vinci_core.agent.iomt_agent import iomt_agent
from vinci_core.agent.genomics_agent import pharmacogenomics_agent
from vinci_core.agent.regulatory_agent import regulatory_copilot
from vinci_core.agent.patient_agent import patient_agent

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
        ],
    }
