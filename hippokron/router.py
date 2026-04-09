from fastapi import APIRouter
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

from .service import clinical_query
from vinci_core.schemas import AIResponse
from vinci_core.workflows.clinical import optimize_trial_protocol, match_patients

router = APIRouter(prefix="/hippokron", tags=["HippoKron — Clinical"])


class ClinicalRequest(BaseModel):
    prompt: str
    patient_id: str | None = None
    patient_context: dict | None = None


class TrialOptimizeRequest(BaseModel):
    indication: str
    phase: str                            # "1" | "2" | "3" | "4"
    draft_protocol: Optional[str] = None
    patient_population: Optional[Dict[str, Any]] = None


class PatientMatchRequest(BaseModel):
    trial_criteria: Dict[str, Any]
    patient_records: List[Dict[str, Any]]


@router.post("/query", response_model=AIResponse)
async def query(request: ClinicalRequest):
    return await clinical_query(request.prompt, request.patient_context)


@router.post("/trials/optimize")
async def trial_optimize(request: TrialOptimizeRequest):
    """
    Optimize a clinical trial protocol.

    Analyzes the draft protocol against active comparable trials
    (via ClinicalTrials.gov) and returns evidence-grounded recommendations.

    Example:
      POST /api/v1/hippokron/trials/optimize
      { "indication": "NSCLC", "phase": "2", "draft_protocol": "..." }
    """
    return await optimize_trial_protocol(
        indication=request.indication,
        phase=request.phase,
        draft_protocol=request.draft_protocol,
        patient_population=request.patient_population,
    )


@router.post("/trials/match-patients")
async def trial_match_patients(request: PatientMatchRequest):
    """
    Match patient records to trial eligibility criteria.

    Evaluates each patient as ELIGIBLE / INELIGIBLE / BORDERLINE
    with rationale and screening risk flags.

    Example:
      POST /api/v1/hippokron/trials/match-patients
      {
        "trial_criteria": { "age_range": "18-65", "diagnosis": "NSCLC stage III-IV", "ecog": "0-1" },
        "patient_records": [{ "patient_id": "pt-001", "age": 55, "diagnosis": "NSCLC", "ecog": 1 }]
      }
    """
    return await match_patients(
        trial_criteria=request.trial_criteria,
        patient_records=request.patient_records,
    )


@router.get("/health")
async def health():
    return {"status": "ok", "layer": "hippokron"}
