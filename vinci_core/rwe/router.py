"""
Real-World Evidence API — Phase 2 / Ariston AI.

Endpoints for RWE data licensing, insight generation, and dataset registry.
This is the Phase 2 revenue stream: pharma pays for access to
de-identified, aggregate LATAM health data insights.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from vinci_core.rwe.engine import rwe_engine, RWEDataset

router = APIRouter(prefix="/rwe", tags=["Real-World Evidence (Phase 2)"])


class DatasetRegistrationRequest(BaseModel):
    source_country: str
    therapeutic_area: str
    record_count: int
    date_range_start: str
    date_range_end: str
    access_tier: str = "aggregate"


class InsightRequest(BaseModel):
    therapeutic_area: str
    countries: list[str]
    research_question: str
    license_tier: str = "standard"


class LicensingProposalRequest(BaseModel):
    company_name: str
    therapeutic_areas: list[str]
    countries: list[str]
    tier: str = Field("standard", description="standard | premium | exclusive")


@router.post("/datasets/register")
async def register_dataset(req: DatasetRegistrationRequest):
    """Register a new RWE dataset from a LATAM health system partner."""
    import hashlib
    dataset_id = hashlib.sha256(
        f"{req.source_country}{req.therapeutic_area}{req.date_range_start}".encode()
    ).hexdigest()[:12]

    source_info = rwe_engine.get_source_catalog().get(req.source_country.lower(), {})

    dataset = RWEDataset(
        dataset_id=dataset_id,
        source_country=req.source_country.lower(),
        source_name=source_info.get("name", f"{req.source_country.upper()} Health Data"),
        therapeutic_area=req.therapeutic_area,
        record_count=req.record_count,
        date_range_start=req.date_range_start,
        date_range_end=req.date_range_end,
        access_tier=req.access_tier,
    )
    rwe_engine.register_dataset(dataset)
    return {"dataset_id": dataset_id, "status": "registered", "dataset": dataset.to_dict()}


@router.get("/datasets")
async def list_datasets(country: Optional[str] = None, therapeutic_area: Optional[str] = None):
    """List registered RWE datasets with optional filtering."""
    return {"datasets": rwe_engine.list_datasets(country=country, therapeutic_area=therapeutic_area)}


@router.post("/insights/generate")
async def generate_insight(req: InsightRequest):
    """Generate a licensed RWE insight for a research question."""
    insight = rwe_engine.generate_insight(
        therapeutic_area=req.therapeutic_area,
        countries=req.countries,
        research_question=req.research_question,
        license_tier=req.license_tier,
    )
    return {
        "insight_id": insight.insight_id,
        "finding": insight.finding,
        "confidence": insight.confidence,
        "patient_count": insight.patient_count,
        "data_sources": insight.data_sources,
        "license_tier": insight.license_tier,
        "generated_at": insight.generated_at,
    }


@router.post("/licensing/proposal")
async def generate_licensing_proposal(req: LicensingProposalRequest):
    """Generate a data licensing proposal for a pharma partner."""
    proposal = rwe_engine.generate_licensing_proposal(
        company_name=req.company_name,
        therapeutic_areas=req.therapeutic_areas,
        countries=req.countries,
        tier=req.tier,
    )
    return proposal


@router.get("/sources")
async def list_data_sources():
    """List all LATAM health data sources available for RWE partnerships."""
    return {"sources": rwe_engine.get_source_catalog()}
