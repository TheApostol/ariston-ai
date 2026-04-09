from pydantic import BaseModel
from typing import Optional, Dict, Any


class CompletionRequest(BaseModel):
    prompt: str
    model: Optional[str] = "openrouter/free"
    layer: Optional[str] = "base"
    context: Optional[Dict[str, Any]] = None
