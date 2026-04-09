"""
LATAM Regulatory Intelligence API — Ariston AI.

Endpoints for pharmaceutical regulatory submissions across Latin American markets.
Primary go-to-market focus: ANVISA (Brazil), COFEPRIS (Mexico), INVIMA (Colombia),
ANMAT (Argentina), ISP (Chile).

These endpoints are the revenue-generating core of Ariston's Phase 1 LATAM strategy.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from vinci_core.agent.latam_agent import latam_agent, LATAM_AGENCIES
from vinci_core.workflows.pipeline import PipelineContext
from vinci_core.workflows.latam_regulatory_pipeline import latam_pipeline

router = APIRouter(prefix="/latam", tags=["LATAM Regulatory Intelligence"])


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class LatamQueryRequest(BaseModel):
    prompt: str = Field(..., description="Regulatory question or submission request")
    countries: Optional[list[str]] = Field(
        None,
        description="Target LATAM countries: brazil, mexico, colombia, argentina, chile",
    )
    product_description: Optional[str] = Field(None, description="Product being registered")
    existing_approvals: Optional[list[str]] = Field(
        None,
        description="Existing reference approvals (e.g. ['FDA', 'EMA', 'WHO-PQ'])",
    )


class LatamRoadmapRequest(BaseModel):
    countries: list[str] = Field(..., description="Target LATAM markets")
    product_type: str = Field("pharmaceutical", description="pharmaceutical | biologic | medical_device | diagnostic")
    has_fda_approval: bool = False
    has_ema_approval: bool = False


class AgencyProfileResponse(BaseModel):
    country: str
    agency: str
    full_name: str
    language: str
    system: str
    key_regulations: list[str]
    review_days_range: tuple
    priority_pathway: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/query")
async def latam_regulatory_query(request: LatamQueryRequest):
    """
    Run the full LATAM regulatory pipeline for a given prompt.

    Executes: Country Scoping → Gap Analysis → Dossier Strategy → Risk Assessment.
    Returns structured regulatory guidance with per-country submission roadmap.
    """
    prompt = request.prompt
    if request.countries:
        country_str = ", ".join(c.upper() for c in request.countries)
        prompt = f"Target markets: {country_str}.\n{prompt}"

    if request.product_description:
        prompt = f"Product: {request.product_description}\n{prompt}"

    if request.existing_approvals:
        approval_str = ", ".join(request.existing_approvals)
        prompt = f"Existing approvals: {approval_str}.\n{prompt}"

    ctx = PipelineContext(
        prompt=prompt,
        layer="latam",
    )

    try:
        result = await latam_pipeline.run(ctx)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")

    return {
        "content": result.final_content,
        "countries": result.results.get("target_countries", []),
        "roadmap": result.results.get("roadmap", {}),
        "risk_flags": result.results.get("risk_flags", []),
        "metadata": result.metadata,
    }


@router.post("/roadmap")
async def build_latam_roadmap(request: LatamRoadmapRequest):
    """
    Build a multi-country LATAM registration roadmap with timeline estimates.
    Returns: recommended submission sequence, per-country timelines, expedited pathway info.
    """
    roadmap = latam_agent.build_multi_country_roadmap(
        countries=request.countries,
        product_type=request.product_type,
        has_fda_approval=request.has_fda_approval,
        has_ema_approval=request.has_ema_approval,
    )
    return roadmap


@router.get("/agencies")
async def list_agencies():
    """List all supported LATAM regulatory agencies and their profiles."""
    return {
        country: {
            "agency": profile["agency"],
            "full_name": profile["full_name"],
            "language": profile["language"],
            "key_regulations": profile["key_regulations"],
            "review_days_range": profile["review_days_range"],
        }
        for country, profile in LATAM_AGENCIES.items()
    }


@router.get("/agencies/{country}")
async def get_agency_profile(country: str):
    """Get detailed profile for a specific LATAM regulatory agency."""
    profile = latam_agent.get_agency_profile(country.lower())
    if not profile:
        raise HTTPException(
            status_code=404,
            detail=f"Country '{country}' not supported. Valid: {list(LATAM_AGENCIES.keys())}",
        )
    return {"country": country.lower(), **profile}


@router.get("/harmonization")
async def get_harmonization_context():
    """Return PANDRH and MERCOSUR harmonization frameworks relevant to LATAM submissions."""
    return {
        "context": latam_agent.format_harmonization_context(),
        "frameworks": [
            "PANDRH",
            "MERCOSUR GMC Resolutions",
            "ICH adoption by country",
            "WHO Prequalification",
            "IAEA GRP",
        ],
    }
