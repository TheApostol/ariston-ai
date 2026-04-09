from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional

class AIRequest(BaseModel):
    prompt: str
    images: List[str] = Field(default_factory=list)
    fhir_bundle: List[Dict[str, Any]] = Field(default_factory=list)
    context: Dict[str, Any] = Field(default_factory=dict)
    model: Optional[str] = None

class AIResponse(BaseModel):
    model: str
    content: str
    usage: Dict[str, Any]
    metadata: Dict[str, Any]

class JobResponse(BaseModel):
    job_id: str
    status: str
