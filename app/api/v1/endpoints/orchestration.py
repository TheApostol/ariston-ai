from fastapi import APIRouter, BackgroundTasks, WebSocket, WebSocketDisconnect
from app.schemas.orchestration import AIRequest, AIResponse, JobResponse
from app.core.websocket import manager
from vinci_core.engine import engine
from vinci_core.workflows.clinical_pipeline import ClinicalPipeline
from vinci_core.logger import audit_logger
import uuid
import logging
import asyncio

router = APIRouter()

async def background_clinical_job(request: AIRequest, job_id: str):
    try:
        # Broadcast Start
        await manager.broadcast_job_update(job_id, "processing")
        
        # Unified Iterative Execution
        response = await engine.run(request)
        
        # Log to Audit DB
        audit_logger.log_event(
            job_id=job_id,
            intent=request.context.get("layer", "clinical"),
            model=response.model,
            prompt=request.prompt,
            response=response.content,
            score=response.metadata.get("grounded_entities_count", 0),
            meta=response.metadata
        )
        
        # ⚖️ GxP Audit Ledger Transaction
        from app.services.audit_ledger import AristonAuditLedger
        AristonAuditLedger.log_decision(
            job_id=job_id,
            prompt=request.prompt,
            result=response.content,
            metadata=response.metadata
        )
        
        # Broadcast Completion
        await manager.broadcast_job_update(job_id, "completed", {
            "content": response.content,
            "metadata": response.metadata
        })
        
    except Exception as e:
        logging.error(f"Job {job_id} Failed: {str(e)}")
        audit_logger.log_event(
            job_id=job_id,
            intent=request.context.get("layer", "clinical"),
            model="error",
            prompt=request.prompt,
            response=str(e),
            safety_violation=True
        )
        await manager.broadcast_job_update(job_id, "failed", {"error": str(e)})

@router.post("/orchestrate", response_model=JobResponse)
async def orchestrate(request: AIRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    background_tasks.add_task(background_clinical_job, request, job_id)
    return JobResponse(job_id=job_id, status="accepted")

@router.websocket("/ws/jobs/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(client_id, websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle heartbeat or client messages if needed
    except WebSocketDisconnect:
        manager.disconnect(client_id)
@router.get("/audit")
async def get_audit_trail():
    from app.services.audit_ledger import AristonAuditLedger
    return AristonAuditLedger.get_audit_trail()

@router.get("/patient/{patient_id}/history")
async def get_patient_history(patient_id: str):
    from vinci_core.agent.patient_agent import patient_agent
    return {"history": patient_agent.get_full_history(patient_id)}

@router.post("/patient/{patient_id}/record")
async def add_patient_record(patient_id: str, event_type: str, details: str):
    from vinci_core.agent.patient_agent import patient_agent
    patient_agent.add_record(patient_id, event_type, details)
    return {"status": "success"}
