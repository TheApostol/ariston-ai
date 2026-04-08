from fastapi import APIRouter
from vinci_core.engine import engine
from vinci_core.schemas import AIRequest, AIResponse

router = APIRouter(prefix="/vinci", tags=["Vinci Core"])


@router.get("/models")
async def list_models():
    return {"models": engine.available_models}


@router.post("/complete", response_model=AIResponse)
async def complete(request: AIRequest):
    return await engine.run(request)
