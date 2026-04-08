from fastapi import APIRouter
from pydantic import BaseModel
from .service import pharma_query
from vinci_core.schemas import AIResponse

router = APIRouter(prefix="/pharma", tags=["Ariston Pharma — Regulatory"])


class PharmaRequest(BaseModel):
    prompt: str
    drug_context: dict | None = None


@router.post("/regulatory")
async def run_regulatory(prompt: str):
    return await pharma_query(prompt)


@router.get("/health")
async def health():
    return {"status": "ok", "layer": "ariston_pharma"}
