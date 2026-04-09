from pydantic import BaseModel, field_validator
from typing import Optional, Dict, Any

VALID_LAYERS = {"base", "pharma", "clinical", "data", "radiology", "general"}
VALID_MODELS = {"openrouter/free", "openrouter", "anthropic", "ollama", "consensus"}


class CompletionRequest(BaseModel):
    prompt: str
    model: Optional[str] = "openrouter/free"
    layer: Optional[str] = "base"
    context: Optional[Dict[str, Any]] = None

    @field_validator("layer")
    @classmethod
    def validate_layer(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in VALID_LAYERS:
            raise ValueError(
                f"Invalid layer '{v}'. Must be one of: {sorted(VALID_LAYERS)}"
            )
        return v

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in VALID_MODELS:
            raise ValueError(
                f"Invalid model '{v}'. Must be one of: {sorted(VALID_MODELS)}"
            )
        return v
