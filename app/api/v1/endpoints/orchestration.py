"""
Primary orchestration API.
- POST /orchestrate — submit a job (background, non-blocking)
- WebSocket /ws/jobs/{client_id} — real-time status
- GET /health — provider health probe
- GET /audit — GxP audit trail
- GET /patient/{id}/history — longitudinal records
- POST /patient/{id}/record — add patient event
- GET /benchmarks — MedPerf eval logs
"""

import asyncio
import os
import time
import uuid
import json
import logging
import httpx
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
        logger.error(
            '{"event":"job_failed","job_id":"%s","error_type":"%s"}',
            job_id, type(e).__name__,
        )
        await manager.broadcast(job_id, "failed", {"error": "Job processing failed. Please try again."})


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


@router.get("/health")
async def health_check():
    """
    Probe AI provider availability.

    Concurrently checks Anthropic, OpenRouter, and Ollama (5 s timeout each).
    Always returns HTTP 200; degraded state is reflected in the body.
    """
    _TIMEOUT = 5.0

    async def _check_anthropic() -> str:
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            return "no_key"
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                r = await client.get(
                    "https://api.anthropic.com/v1/models",
                    headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
                )
            return "ok" if r.status_code < 500 else "error"
        except Exception:
            return "unreachable"

    async def _check_openrouter() -> str:
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        if not api_key:
            return "no_key"
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                r = await client.get(
                    "https://openrouter.ai/api/v1/models",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
            return "ok" if r.status_code < 500 else "error"
        except Exception:
            return "unreachable"

    async def _check_ollama() -> str:
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                r = await client.get(f"{base_url}/api/tags")
            return "ok" if r.status_code < 500 else "error"
        except Exception:
            return "unreachable"

    t0 = time.monotonic()
    anthropic_status, openrouter_status, ollama_status = await asyncio.gather(
        _check_anthropic(), _check_openrouter(), _check_ollama()
    )
    latency_ms = round((time.monotonic() - t0) * 1000)

    providers = {
        "anthropic": anthropic_status,
        "openrouter": openrouter_status,
        "ollama": ollama_status,
    }
    overall = "ok" if all(v in ("ok", "no_key") for v in providers.values()) else "degraded"

    return {
        "status": overall,
        "providers": providers,
        "latency_ms": latency_ms,
    }


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
