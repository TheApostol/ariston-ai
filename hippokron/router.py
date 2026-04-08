from fastapi import APIRouter
from pydantic import BaseModel
from .service import clinical_query
from vinci_core.schemas import AIResponse

router = APIRouter(prefix="/hippokron", tags=["HippoKron — Clinical"])


class ClinicalRequest(BaseModel):
    prompt: str
    patient_id: str | None = None
    patient_context: dict | None = None


@router.post("/query", response_model=AIResponse)
async def query(request: ClinicalRequest):
    return await clinical_query(request.prompt, request.patient_context)


@router.get("/health")
async def health():
    return {"status": "ok", "layer": "hippokron"}
