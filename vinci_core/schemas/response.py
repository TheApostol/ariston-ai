from pydantic import BaseModel
from typing import Optional, Dict, Any


class AIResponse(BaseModel):
    model: str
    content: str
    usage: Optional[Dict[str, int]] = None
    metadata: Optional[Dict[str, Any]] = None
