from pydantic import BaseModel
from typing import Any


class AIRequest(BaseModel):
    prompt: str
    model: str | None = None
    context: dict[str, Any] | None = None
    stream: bool = False


class AIResponse(BaseModel):
    model: str
    content: str
    usage: dict[str, int] | None = None
    metadata: dict[str, Any] | None = None
