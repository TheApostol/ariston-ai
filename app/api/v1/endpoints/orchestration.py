"""
Primary orchestration API.
- POST /orchestrate — submit a job (background, non-blocking)
- WebSocket /ws/jobs/{client_id} — real-time status
- GET /audit — GxP audit trail
- GET /patient/{id}/history — longitudinal records
- POST /patient/{id}/record — add patient event
- GET /benchmarks — MedPerf eval logs
"""

import uuid
import json
import logging
from fastapi import APIRouter, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from vinci_core.schemas import AIResponse
from app.schemas.orchestration import OrchestrateRequest, JobResponse
from app.core.websocket import manager
from app.services.audit_ledger import AristonAuditLedger
from vinci_core.engine import engine
from vinci_core.engine_stream import stream_response
from vinci_core.agent.patient_agent import patient_agent

router = APIRouter()
logger = logging.getLogger("ariston.api")


async def _run_job(request: OrchestrateRequest, job_id: str):
    try:
        await manager.broadcast(job_id, "processing")

        response: AIResponse = await engine.run(
            prompt=request.prompt,
            model=request.model,
            layer=request.layer,
            context=request.context,
            use_rag=request.use_rag,
            patient_id=request.patient_id,
        )

        await manager.broadcast(job_id, "completed", {
            "content": response.content,
            "model": response.model,
            "metadata": response.metadata,
        })

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        await manager.broadcast(job_id, "failed", {"error": str(e)})


@router.post("/orchestrate", response_model=JobResponse)
async def orchestrate(request: OrchestrateRequest, background_tasks: BackgroundTasks):
    """Submit a Life Sciences AI job. Returns job_id immediately; result delivered via WebSocket."""
    job_id = str(uuid.uuid4())
    background_tasks.add_task(_run_job, request, job_id)
    return JobResponse(job_id=job_id, status="accepted")


@router.websocket("/ws/jobs/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(client_id, websocket)
    try:
        while True:
            await websocket.receive_text()  # heartbeat / keep-alive
    except WebSocketDisconnect:
        manager.disconnect(client_id)


@router.get("/audit")
async def get_audit_trail():
    """Retrieve the GxP-compliant audit ledger."""
    return AristonAuditLedger.get_audit_trail()


@router.get("/patient/{patient_id}/history")
async def get_patient_history(patient_id: str):
    return {"patient_id": patient_id, "history": patient_agent.get_full_history(patient_id)}


@router.post("/patient/{patient_id}/record")
async def add_patient_record(patient_id: str, event_type: str, details: str, date: str = None):
    patient_agent.add_record(patient_id, event_type, details, date)
    return {"status": "ok"}


@router.post("/stream")
async def stream(request: OrchestrateRequest):
    """
    Streaming endpoint — returns tokens as Server-Sent Events.
    Connect with: EventSource('/api/v1/stream') or fetch with stream: true.
    """
    async def generate():
        async for chunk in stream_response(
            prompt=request.prompt,
            layer=request.layer,
            context=request.context,
            use_rag=request.use_rag,
            patient_id=request.patient_id,
        ):
            yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.get("/benchmarks")
async def get_benchmarks():
    """Return MedPerf evaluation logs."""
    try:
        with open("benchmarks/eval_logs.jsonl") as f:
            return [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        return []


@router.post("/benchmarks/run")
async def run_benchmark(benchmark: str = "medqa", n: int = 20, background_tasks: BackgroundTasks = None):
    """Run MedPerf benchmark against the engine. Returns job_id; results saved to benchmarks/."""
    from vinci_core.evaluation.medperf import run_benchmark as _run
    job_id = str(uuid.uuid4())
    background_tasks.add_task(_run, benchmark, n)
    return {"job_id": job_id, "benchmark": benchmark, "n": n, "status": "running"}
