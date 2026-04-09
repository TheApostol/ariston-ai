from fastapi import APIRouter
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

from .service import data_query
from vinci_core.workflows.data import detect_safety_signals, generate_rwe_insights

router = APIRouter()


class DataRequest(BaseModel):
    prompt: str
    dataset: Optional[Dict[str, Any]] = None
    signals: Optional[Dict[str, Any]] = None


class SafetySignalRequest(BaseModel):
    drug_name: str
    dataset_summary: Optional[Dict[str, Any]] = None
    time_period: Optional[str] = None


class RWERequest(BaseModel):
    research_question: str
    dataset_description: Optional[str] = None
    data_sources: Optional[List[str]] = None


@router.post("/predict")
async def run_prediction(request: DataRequest):
    return await data_query(request.prompt, request.dataset, request.signals)


@router.post("/signals")
async def pharmacovigilance_signals(request: SafetySignalRequest):
    """
    Detect pharmacovigilance safety signals for a drug.

    Combines FAERS adverse event data with literature evidence
    to surface emerging safety concerns.

    Example:
      POST /api/v1/darwina/signals
      { "drug_name": "semaglutide", "time_period": "2022-2024" }
    """
    return await detect_safety_signals(
        drug_name=request.drug_name,
        dataset_summary=request.dataset_summary,
        time_period=request.time_period,
    )


@router.post("/rwe")
async def rwe_insights(request: RWERequest):
    """
    Generate real-world evidence insights from a research question.

    Returns key findings, effect sizes with confidence intervals,
    confounders, and regulatory/clinical implications.

    Example:
      POST /api/v1/darwina/rwe
      {
        "research_question": "Does GLP-1 receptor agonist use reduce cardiovascular events in T2DM?",
        "data_sources": ["US claims data", "EHR network"]
      }
    """
    return await generate_rwe_insights(
        research_question=request.research_question,
        dataset_description=request.dataset_description,
        data_sources=request.data_sources,
    )


@router.get("/health")
async def health():
    return {"status": "ok", "layer": "darwina"}
