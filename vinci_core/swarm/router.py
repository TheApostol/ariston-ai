"""
Agent Swarm — FastAPI Router.

Endpoints:
  POST /swarm/run         — execute the full multi-agent swarm
  POST /swarm/run/async   — submit swarm job (background, non-blocking)
  GET  /swarm/health      — health check
"""

import uuid
from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

from vinci_core.swarm import agent_swarm
from app.core.websocket import manager

router = APIRouter(prefix="/swarm", tags=["Agent Swarm"])


# ── Request model ─────────────────────────────────────────────────────────────

class SwarmRequest(BaseModel):
    prompt: str
    patient_id: Optional[str] = None
    drug_name: Optional[str] = None
    fhir_bundle: Optional[List[Dict[str, Any]]] = None
    telemetry: Optional[Dict[str, Any]] = None
    genetics: Optional[List[str]] = None
    include_stages: Optional[List[str]] = None   # default: all stages


# ── Synchronous swarm ─────────────────────────────────────────────────────────

@router.post("/run")
async def run_swarm(request: SwarmRequest):
    """
    Execute the full Ariston Agent Swarm synchronously.

    Chains all agents in order:
      Patient History → Intent Classifier → PGx → Digital Twin →
      IoMT Adherence → Clinical Pipeline → Regulatory Copilot

    Each stage enriches the context for subsequent agents.

    Example:
      POST /api/v1/swarm/run
      {
        "prompt": "Patient presents with chest pain and is on clopidogrel",
        "patient_id": "pt-001",
        "drug_name": "clopidogrel",
        "genetics": ["CYP2C19 Poor Metabolizer"]
      }
    """
    return await agent_swarm.run(
        prompt=request.prompt,
        patient_id=request.patient_id,
        drug_name=request.drug_name,
        fhir_bundle=request.fhir_bundle,
        telemetry=request.telemetry,
        genetics=request.genetics,
        include_stages=request.include_stages,
    )


# ── Async (background) swarm ──────────────────────────────────────────────────

async def _run_swarm_background(request: SwarmRequest, job_id: str):
    try:
        await manager.broadcast(job_id, "processing")
        result = await agent_swarm.run(
            prompt=request.prompt,
            patient_id=request.patient_id,
            drug_name=request.drug_name,
            fhir_bundle=request.fhir_bundle,
            telemetry=request.telemetry,
            genetics=request.genetics,
            include_stages=request.include_stages,
        )
        await manager.broadcast(job_id, "completed", {"swarm_result": result})
    except Exception as e:
        await manager.broadcast(job_id, "failed", {"error": str(e)})


@router.post("/run/async")
async def run_swarm_async(request: SwarmRequest, background_tasks: BackgroundTasks):
    """
    Submit a swarm job asynchronously.

    Returns job_id immediately; results delivered via WebSocket at
    `/api/v1/ws/jobs/{job_id}`.
    """
    job_id = str(uuid.uuid4())
    background_tasks.add_task(_run_swarm_background, request, job_id)
    return {
        "job_id": job_id,
        "status": "accepted",
        "websocket": f"/api/v1/ws/jobs/{job_id}",
    }


@router.get("/health")
async def health():
    return {
        "status": "ok",
        "layer": "swarm",
        "agents": [
            "patient_history", "intent_classifier", "pharmacogenomics",
            "pharmacist", "digital_twin", "iomt",
            "clinical_pipeline", "regulatory_copilot",
        ],
    }
