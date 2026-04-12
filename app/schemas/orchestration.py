from pydantic import BaseModel
from typing import Optional, Dict, Any, List


class OrchestrateRequest(BaseModel):
    prompt: str
    layer: Optional[str] = None           # auto-detected if not provided
    model: Optional[str] = None           # use layer default if not provided
    patient_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    use_rag: bool = True
    images: Optional[List[str]] = None    # base64-encoded images (X-rays, CT, MRI, etc.)
    fhir_bundle: Optional[List[Dict[str, Any]]] = None   # FHIR R4 resources
    documents: Optional[List[Dict[str, Any]]] = None     # { name, content, type }


class JobResponse(BaseModel):
    job_id: str
    status: str
