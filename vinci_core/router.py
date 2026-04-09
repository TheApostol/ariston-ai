import logging

from fastapi import APIRouter, HTTPException
from vinci_core.engine import engine
from vinci_core.schemas import CompletionRequest, AIResponse

logger = logging.getLogger("ariston.api.vinci")
router = APIRouter()


@router.post(
    "/vinci/complete",
    response_model=AIResponse,
    summary="Ariston AI completion endpoint",
    description=(
        "Runs a prompt through Ariston AI using the selected model and layer. "
        "Supports local (Ollama) and free cloud (OpenRouter) inference."
    ),
)
async def complete(request: CompletionRequest):
    """
    Request body:
    {
        "prompt": "...",
        "model": "openrouter/free" | "ollama",
        "layer": "base" | "pharma" | "clinical" | "data" | "radiology" | "general",
        "context": {}
    }
    """
    try:
        result = await engine.run(**request.model_dump())
        return result
    except Exception as e:
        logger.error(
            '{"event":"api_error","endpoint":"/vinci/complete","error_type":"%s"}',
            type(e).__name__,
        )
        raise HTTPException(
            status_code=500,
            detail="An unexpected server error occurred. Please try again.",
        )
