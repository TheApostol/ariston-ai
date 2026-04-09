from pydantic import BaseModel
from typing import Optional, Dict, Any


class OrchestrateRequest(BaseModel):
    prompt: str
    layer: Optional[str] = None          # auto-detected if not provided
    model: Optional[str] = None          # use layer default if not provided
    patient_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    use_rag: bool = True


class JobResponse(BaseModel):
    job_id: str
    status: str
