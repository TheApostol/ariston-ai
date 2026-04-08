from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from vinci_core.engine import engine
from vinci_core.schemas import AIRequest
from vinci_core.workflows.clinical_pipeline import ClinicalPipeline
import logging

app = FastAPI(title="Ariston AI Event Bus", version="1.0.0")

# Background worker queue for async execution
async def background_clinical_job(request: AIRequest, job_id: str):
    try:
        pipeline = ClinicalPipeline(engine)
        response = await pipeline.execute(request)
        logging.info(f"Job {job_id} Completed. Focus: {response.model}")
        # In a real environment, this would publish the response to Kafka/WebSockets here.
    except Exception as e:
        logging.error(f"Job {job_id} Failed: {str(e)}")

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
