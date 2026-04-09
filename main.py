from fastapi import FastAPI, BackgroundTasks, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from vinci_core.engine import engine
from vinci_core.schemas import AIRequest
from vinci_core.workflows.clinical_pipeline import ClinicalPipeline
import logging
import os

app = FastAPI(title="Ariston AI Event Bus", version="1.0.0")

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_index():
    return FileResponse('static/index.html')

# Background worker queue for async execution
async def background_clinical_job(request: AIRequest, job_id: str):
    from vinci_core.logger import audit_logger
    try:
        pipeline = ClinicalPipeline(engine)
        response = await pipeline.execute(request)
        logging.info(f"Job {job_id} Completed. Focus: {response.model}")
        
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

class JobResponse(BaseModel):
    job_id: str
    status: str

@app.post("/api/v1/orchestrate", response_model=JobResponse)
async def orchestrate_request(request: AIRequest, background_tasks: BackgroundTasks):
    """
    Accepts clinical requests and drops them onto the async background event bus
    for processing without blocking the API gateway.
    """
    import uuid
    job_id = str(uuid.uuid4())
    
    background_tasks.add_task(background_clinical_job, request, job_id)
    
    return JobResponse(job_id=job_id, status="accepted_for_processing")

@app.get("/health")
def health_check():
    return {"status": "healthy", "engine": "vinci-orchestrator-active"}
