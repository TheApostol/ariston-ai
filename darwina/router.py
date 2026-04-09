from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional, Dict, Any
from .service import data_query

router = APIRouter()


class DataRequest(BaseModel):
    prompt: str
    dataset: Optional[Dict[str, Any]] = None
    signals: Optional[Dict[str, Any]] = None


@router.post("/predict")
async def run_prediction(request: DataRequest):
    return await data_query(request.prompt, request.dataset, request.signals)


@router.get("/health")
async def health():
    return {"status": "ok", "layer": "darwina"}
