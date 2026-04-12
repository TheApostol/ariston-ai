"""
Primary orchestration API.
- POST /orchestrate — submit a job (background, non-blocking)
- POST /analyze    — synchronous full analysis (images + documents + prompt)
- POST /upload     — multipart file upload → base64 → analysis
- WebSocket /ws/jobs/{client_id} — real-time status
- GET /health — provider health probe
- GET /audit — GxP audit trail
- GET /patient/{id}/history — longitudinal records
- POST /patient/{id}/record — add patient event
- GET /benchmarks — MedPerf eval logs
"""

import asyncio
import base64
import os
import time
import uuid
import json
import logging
import httpx
from fastapi import APIRouter, BackgroundTasks, WebSocket, WebSocketDisconnect, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from typing import List, Optional
from vinci_core.schemas import AIResponse
from app.schemas.orchestration import OrchestrateRequest, JobResponse
from app.core.websocket import manager
from app.services.audit_ledger import AristonAuditLedger
from vinci_core.engine import engine
from vinci_core.engine_stream import stream_response
from vinci_core.agent.patient_agent import patient_agent
from vinci_core.agent.vision_agent import VisionRadiologyAgent

router = APIRouter()
logger = logging.getLogger("ariston.api")

_vision_agent = VisionRadiologyAgent()

# MIME types that are treated as images for vision analysis
_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif", "image/dicom"}
# Document types for text extraction + RAG
_DOC_TYPES = {"application/pdf", "application/json", "text/plain", "text/csv"}


async def _run_job(request: OrchestrateRequest, job_id: str):
    try:
        await manager.broadcast(job_id, "processing")

        result_data = {}

        # --- Vision analysis if images present ---
        images = request.images or []
        if images:
            await manager.broadcast(job_id, "processing", {"stage": "vision_analysis"})
            vision_result = await _vision_agent.analyze_scan(
                prompt=request.prompt,
                images=images,
            )
            result_data["vision_analysis"] = vision_result

        # --- Inject FHIR + document context ---
        context = dict(request.context or {})
        if request.fhir_bundle:
            context["fhir_bundle"] = request.fhir_bundle
        if request.documents:
            context["documents"] = [
                f"[{d.get('name', 'doc')}]: {d.get('content', '')[:2000]}"
                for d in request.documents
            ]

        # --- Core engine (LLM + RAG + safety) ---
        layer = request.layer
        if images and not layer:
            layer = "radiology"   # auto-route image requests to radiology layer

        response: AIResponse = await engine.run(
            prompt=request.prompt,
            model=request.model,
            layer=layer,
            context=context,
            use_rag=request.use_rag,
            patient_id=request.patient_id,
        )

        result_data.update({
            "content": response.content,
            "model": response.model,
            "metadata": response.metadata,
        })

        await manager.broadcast(job_id, "completed", result_data)

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


@router.post("/upload")
async def upload_files(
    files: List[UploadFile] = File(...),
    prompt: str = Form(default="Analyze this file and provide a full clinical report."),
    patient_id: Optional[str] = Form(default=None),
    layer: Optional[str] = Form(default=None),
):
    """
    Multipart file upload endpoint.
    Accepts: images (JPEG, PNG, DICOM), PDFs, FHIR JSON, lab reports, CSV.
    Returns: synchronous full analysis — vision report + LLM + RAG.

    Example (curl):
      curl -X POST /api/v1/upload \\
        -F "files=@xray.jpg" \\
        -F "prompt=Analyze this chest X-ray for pneumonia" \\
        -F "patient_id=ARISTON-001"
    """
    images = []
    documents = []

    for f in files:
        raw = await f.read()
        mime = f.content_type or "application/octet-stream"
        name = f.filename or "uploaded_file"

        if mime in _IMAGE_TYPES or name.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".dcm")):
            b64 = base64.b64encode(raw).decode()
            images.append(f"data:{mime};base64,{b64}")
            logger.info("[upload] image file=%s mime=%s size=%d", name, mime, len(raw))

        elif mime == "application/json" or name.endswith(".json"):
            try:
                content = raw.decode("utf-8")
                documents.append({"name": name, "content": content, "type": "json"})
            except Exception:
                pass

        elif mime == "text/plain" or name.endswith((".txt", ".csv")):
            content = raw.decode("utf-8", errors="replace")
            documents.append({"name": name, "content": content, "type": "text"})

        elif mime == "application/pdf" or name.endswith(".pdf"):
            # Extract text from PDF if pypdf available, else store as placeholder
            try:
                import io
                from pypdf import PdfReader
                reader = PdfReader(io.BytesIO(raw))
                text = "\n".join(p.extract_text() or "" for p in reader.pages)
                documents.append({"name": name, "content": text[:8000], "type": "pdf"})
            except ImportError:
                documents.append({"name": name, "content": f"[PDF: {name} — {len(raw)} bytes]", "type": "pdf"})

        else:
            # Unknown — try UTF-8 decode
            try:
                content = raw.decode("utf-8", errors="replace")[:4000]
                documents.append({"name": name, "content": content, "type": "unknown"})
            except Exception:
                pass

    # Auto-set layer
    inferred_layer = layer
    if not inferred_layer:
        inferred_layer = "radiology" if images else "clinical"

    # Build and run synchronously (for upload flow, user waits for result)
    req = OrchestrateRequest(
        prompt=prompt,
        layer=inferred_layer,
        patient_id=patient_id,
        images=images,
        documents=documents,
        use_rag=True,
    )

    job_id = str(uuid.uuid4())
    result: dict = {}

    # Vision analysis
    if images:
        vision_result = await _vision_agent.analyze_scan(prompt=prompt, images=images)
        result["vision_analysis"] = vision_result

    # Engine analysis
    context: dict = {}
    if documents:
        context["documents"] = [f"[{d['name']}]: {d['content'][:2000]}" for d in documents]

    ai_response: AIResponse = await engine.run(
        prompt=prompt,
        layer=inferred_layer,
        context=context,
        use_rag=True,
        patient_id=patient_id,
    )

    result.update({
        "job_id": job_id,
        "content": ai_response.content,
        "model": ai_response.model,
        "metadata": ai_response.metadata,
        "files_processed": [f.filename for f in files],
        "images_analyzed": len(images),
        "documents_extracted": len(documents),
        "layer": inferred_layer,
    })

    return result


@router.post("/analyze")
async def analyze(request: OrchestrateRequest):
    """
    Synchronous full analysis — blocks until complete.
    Supports images (base64) + documents + prompt.
    Use /orchestrate for async (WebSocket) flow.
    Use /upload for multipart file uploads.
    """
    job_id = str(uuid.uuid4())
    result: dict = {}

    images = request.images or []
    if images:
        vision_result = await _vision_agent.analyze_scan(
            prompt=request.prompt,
            images=images,
        )
        result["vision_analysis"] = vision_result

    context = dict(request.context or {})
    if request.fhir_bundle:
        context["fhir_bundle"] = request.fhir_bundle
    if request.documents:
        context["documents"] = [
            f"[{d.get('name', 'doc')}]: {d.get('content', '')[:2000]}"
            for d in request.documents
        ]

    layer = request.layer
    if images and not layer:
        layer = "radiology"

    ai_response: AIResponse = await engine.run(
        prompt=request.prompt,
        model=request.model,
        layer=layer,
        context=context,
        use_rag=request.use_rag,
        patient_id=request.patient_id,
    )

    result.update({
        "job_id": job_id,
        "content": ai_response.content,
        "model": ai_response.model,
        "metadata": ai_response.metadata,
        "images_analyzed": len(images),
        "layer": layer,
    })

    return result


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
