from fastapi import APIRouter, HTTPException
from vinci_core.engine import engine
from vinci_core.schemas import CompletionRequest, AIResponse


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
        "layer": "base" | "pharma",
        "context": {}
    }
    """

    try:
        result = await engine.run(**request.dict())
        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )