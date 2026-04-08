from pydantic import BaseModel, Field
from typing import Any, Literal

class AIRequest(BaseModel):
    prompt: str = Field(..., description="The user's query or prompt.")
    model: str | None = Field(default=None, description="Optional forced model override.")
    context: dict[str, Any] | None = Field(default_factory=dict, description="Metadata and history context.")
    stream: bool = False

class ClinicalRequest(AIRequest):
    """Payload specifically for Clinical diagnostic intents"""
    patient_age: int | None = None
    patient_sex: Literal["M", "F", "Other"] | None = None
    symptoms: list[str] = Field(default_factory=list)

class PharmaRequest(AIRequest):
    """Payload specifically for Pharmacological intents"""
    drugs_mentioned: list[str] = Field(default_factory=list)
    check_interactions: bool = True

class DataRequest(AIRequest):
    """Payload specifically for Data parsing intents"""
    extract_entities: bool = True

class AIResponse(BaseModel):
    model: str
    content: str
    usage: dict[str, int] | None = None
    metadata: dict[str, Any] | None = None
