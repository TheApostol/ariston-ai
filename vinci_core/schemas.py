from pydantic import BaseModel, Field
from typing import Any, Literal

class FHIRResource(BaseModel):
    resourceType: str
    id: str | None = None
    
class FHIRObservation(FHIRResource):
    status: str
    code: dict[str, Any]
    valueQuantity: dict[str, Any] | None = None
    valueString: str | None = None

class AIRequest(BaseModel):
    prompt: str = Field(..., description="The user's query or prompt.")
    patient_id: str | None = Field(default=None, description="Longitudinal patient identifier for GxP history tracking.")
    model: str | None = Field(default=None, description="Optional forced model override.")
    context: dict[str, Any] | None = Field(default_factory=dict, description="Metadata and history context.")
    fhir_bundle: list[dict[str, Any]] | None = Field(default=None, description="Optional raw FHIR API inputs")
    images: list[str] | None = Field(default=None, description="List of Base64 strings or URLs for medical scans.")
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
