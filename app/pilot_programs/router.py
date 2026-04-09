"""
Pilot Program — FastAPI Router.

Endpoints:
  POST /pilots/enroll                   — enroll new pilot (<5 min)
  GET  /pilots                          — list all pilots
  GET  /pilots/{id}                     — get pilot details
  PATCH /pilots/{id}/status             — update pilot status
  GET  /pilots/analytics                — platform-wide analytics

  POST /pilots/{id}/documents           — save document version
  GET  /pilots/{id}/documents           — list document versions
  GET  /pilots/{id}/documents/{doc_id}  — get document content

  POST /pilots/{id}/roi                 — record ROI metric
  GET  /pilots/{id}/roi                 — get ROI summary

  POST /pilots/{id}/feedback            — submit pilot feedback
  GET  /pilots/{id}/feedback            — list pilot feedback

  GET  /pilots/health                   — health check
"""

from fastapi import APIRouter, Query
from pydantic import BaseModel, EmailStr
from typing import Optional

from .service import (
    enroll_pilot,
    get_pilot,
    list_pilots,
    update_pilot_status,
    save_document_version,
    get_document_versions,
    get_document_content,
    record_roi_metric,
    get_roi_summary,
    submit_pilot_feedback,
    get_pilot_feedback,
    get_all_pilots_analytics,
)

router = APIRouter(prefix="/pilots", tags=["LatAm Pilot Programs"])


# ── Request models ────────────────────────────────────────────────────────────

class EnrollPilotRequest(BaseModel):
    company_name: str
    contact_name: str
    contact_email: str
    country: str                          # Brazil | Mexico | Colombia | Argentina
    locale: str                           # pt-BR | es-MX | es-CO | es-AR
    agency: str                           # ANVISA | COFEPRIS | INVIMA | ANMAT
    therapeutic_area: str                 # oncology | rare_disease | cardiology ...
    commitment_level: str                 # trial | committed | enterprise
    notes: Optional[str] = None
    metadata: Optional[dict] = None


class UpdateStatusRequest(BaseModel):
    status: str                           # active | paused | completed


class SaveDocumentRequest(BaseModel):
    document_type: str                    # csr | ectd | anvisa_registro | etc.
    content: str
    drug_name: Optional[str] = None
    indication: Optional[str] = None
    language: str = "en"
    agency: Optional[str] = None
    created_by: Optional[str] = None
    change_summary: Optional[str] = None


class ROIMetricRequest(BaseModel):
    document_type: str
    manual_hours_baseline: float          # hours to complete without AI
    ai_assisted_hours: float              # hours with AI assistance
    hourly_rate_usd: float = 150.0        # regulatory writer hourly rate
    documents_generated: int = 1
    user_sessions: int = 1
    metadata: Optional[dict] = None


class PilotFeedbackRequest(BaseModel):
    rating: Optional[int] = None          # 1-5
    nps_score: Optional[int] = None       # 0-10
    feature_ratings: Optional[dict] = None
    comment: Optional[str] = None
    blockers: Optional[str] = None
    feature_requests: Optional[str] = None


# ── Pilot enrollment ──────────────────────────────────────────────────────────

@router.post("/enroll")
async def enroll(request: EnrollPilotRequest):
    """
    Enroll a new pharma/biotech pilot organization.

    Can be completed in <5 minutes via this endpoint.

    Example:
      POST /api/v1/pilots/enroll
      {
        "company_name": "BioPharma SA",
        "contact_name": "Dr. Ana González",
        "contact_email": "ana@biopharma.com.mx",
        "country": "Mexico",
        "locale": "es-MX",
        "agency": "COFEPRIS",
        "therapeutic_area": "oncology",
        "commitment_level": "trial"
      }
    """
    return enroll_pilot(
        company_name=request.company_name,
        contact_name=request.contact_name,
        contact_email=request.contact_email,
        country=request.country,
        locale=request.locale,
        agency=request.agency,
        therapeutic_area=request.therapeutic_area,
        commitment_level=request.commitment_level,
        notes=request.notes,
        metadata=request.metadata,
    )


@router.get("")
async def list_all_pilots(
    status: Optional[str] = Query(None, description="Filter by status: active | paused | completed"),
):
    """List all enrolled pilots."""
    pilots = list_pilots(status=status)
    return {
        "pilots": pilots,
        "total": len(pilots),
    }


@router.get("/analytics")
async def platform_analytics():
    """
    Platform-wide analytics dashboard across all pilots.

    Returns aggregate ROI, document generation stats, and adoption metrics.
    """
    return get_all_pilots_analytics()


@router.get("/{pilot_id}")
async def get_pilot_detail(pilot_id: str):
    """Get full details for a specific pilot."""
    pilot = get_pilot(pilot_id)
    if not pilot:
        return {"error": f"Pilot '{pilot_id}' not found"}
    return pilot


@router.patch("/{pilot_id}/status")
async def update_status(pilot_id: str, request: UpdateStatusRequest):
    """Update the status of a pilot (active, paused, completed)."""
    success = update_pilot_status(pilot_id, request.status)
    return {
        "pilot_id": pilot_id,
        "status": request.status,
        "updated": success,
    }


# ── Document versioning ───────────────────────────────────────────────────────

@router.post("/{pilot_id}/documents")
async def save_document(pilot_id: str, request: SaveDocumentRequest):
    """
    Save a new document version for a pilot.

    Each save auto-increments version number.
    Previous versions are retained for audit/rollback (A/B testing).

    Example:
      POST /api/v1/pilots/{pilot_id}/documents
      { "document_type": "csr", "content": "...", "drug_name": "dabrafenib", "change_summary": "Updated Section 12" }
    """
    pilot = get_pilot(pilot_id)
    if not pilot:
        return {"error": f"Pilot '{pilot_id}' not found"}

    return save_document_version(
        pilot_id=pilot_id,
        document_type=request.document_type,
        content=request.content,
        drug_name=request.drug_name,
        indication=request.indication,
        language=request.language,
        agency=request.agency,
        created_by=request.created_by,
        change_summary=request.change_summary,
    )


@router.get("/{pilot_id}/documents")
async def list_documents(
    pilot_id: str,
    document_type: Optional[str] = Query(None, description="Filter by document type"),
    active_only: bool = Query(False, description="Return only the current active version"),
):
    """List document versions for a pilot."""
    pilot = get_pilot(pilot_id)
    if not pilot:
        return {"error": f"Pilot '{pilot_id}' not found"}

    versions = get_document_versions(
        pilot_id=pilot_id,
        document_type=document_type,
        active_only=active_only,
    )
    return {
        "pilot_id": pilot_id,
        "documents": versions,
        "total": len(versions),
    }


@router.get("/{pilot_id}/documents/{document_id}")
async def get_document(pilot_id: str, document_id: str):
    """Get the full content of a specific document version."""
    doc = get_document_content(document_id)
    if not doc or doc.get("pilot_id") != pilot_id:
        return {"error": f"Document '{document_id}' not found for pilot '{pilot_id}'"}
    return doc


# ── ROI metrics ───────────────────────────────────────────────────────────────

@router.post("/{pilot_id}/roi")
async def add_roi_metric(pilot_id: str, request: ROIMetricRequest):
    """
    Record an ROI measurement for a pilot.

    Calculates time saved and cost saved (vs. manual baseline).

    Example:
      POST /api/v1/pilots/{pilot_id}/roi
      {
        "document_type": "csr",
        "manual_hours_baseline": 40.0,
        "ai_assisted_hours": 8.0,
        "hourly_rate_usd": 150
      }
    → time_saved: 32h, cost_saved: $4,800, reduction: 80%
    """
    pilot = get_pilot(pilot_id)
    if not pilot:
        return {"error": f"Pilot '{pilot_id}' not found"}

    return record_roi_metric(
        pilot_id=pilot_id,
        document_type=request.document_type,
        manual_hours_baseline=request.manual_hours_baseline,
        ai_assisted_hours=request.ai_assisted_hours,
        hourly_rate_usd=request.hourly_rate_usd,
        documents_generated=request.documents_generated,
        user_sessions=request.user_sessions,
        metadata=request.metadata,
    )


@router.get("/{pilot_id}/roi")
async def get_roi(pilot_id: str):
    """Get aggregated ROI summary for a pilot."""
    pilot = get_pilot(pilot_id)
    if not pilot:
        return {"error": f"Pilot '{pilot_id}' not found"}

    return get_roi_summary(pilot_id)


# ── Pilot feedback ────────────────────────────────────────────────────────────

@router.post("/{pilot_id}/feedback")
async def add_feedback(pilot_id: str, request: PilotFeedbackRequest):
    """
    Submit structured feedback for a pilot.

    Captures NPS, feature ratings, blockers, and feature requests.

    Example:
      POST /api/v1/pilots/{pilot_id}/feedback
      {
        "rating": 4,
        "nps_score": 8,
        "feature_ratings": {"csr_drafting": 5, "regulatory_mapping": 4},
        "blockers": "ANVISA module needs more template sections"
      }
    """
    pilot = get_pilot(pilot_id)
    if not pilot:
        return {"error": f"Pilot '{pilot_id}' not found"}

    return submit_pilot_feedback(
        pilot_id=pilot_id,
        rating=request.rating,
        nps_score=request.nps_score,
        feature_ratings=request.feature_ratings,
        comment=request.comment,
        blockers=request.blockers,
        feature_requests=request.feature_requests,
    )


@router.get("/{pilot_id}/feedback")
async def list_feedback(pilot_id: str):
    """Get all feedback submissions for a pilot."""
    pilot = get_pilot(pilot_id)
    if not pilot:
        return {"error": f"Pilot '{pilot_id}' not found"}

    feedback = get_pilot_feedback(pilot_id)
    return {
        "pilot_id": pilot_id,
        "feedback": feedback,
        "total": len(feedback),
    }


@router.get("/health")
async def health():
    return {"status": "ok", "layer": "pilot_programs"}
