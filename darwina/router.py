from fastapi import APIRouter
from typing import Optional, Dict, Any
from .service import data_query

router = APIRouter()


@router.post("/predict")
async def run_prediction(
    prompt: str,
    dataset_context: Optional[Dict[str, Any]] = None
):
    return await data_query(prompt, dataset_context)


@router.get("/health")
async def health():
    return {"status": "ok", "layer": "darwina"}
