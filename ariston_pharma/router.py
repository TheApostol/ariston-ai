from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from .service import pharma_query
from vinci_core.schemas import AIResponse
from vinci_core.workflows.pharma import draft_regulatory_document, DOCUMENT_TEMPLATES

router = APIRouter(prefix="/pharma", tags=["Ariston Pharma — Regulatory"])


class PharmaRequest(BaseModel):
    prompt: str
    drug_context: Optional[dict] = None


class DraftRequest(BaseModel):
    document_type: str = "csr"        # csr | ectd | cmc | pv_narrative
    drug_name: str
    indication: str
    nct_id: Optional[str] = None      # ClinicalTrials.gov ID for live grounding
    study_data: Optional[dict] = None
    section: Optional[str] = None     # draft a single section only


@router.post("/regulatory", response_model=AIResponse)
async def run_regulatory(request: PharmaRequest):
    """General pharma/regulatory query."""
    return await pharma_query(request.prompt, drug_context=request.drug_context)


@router.post("/draft")
async def draft_document(request: DraftRequest):
    """
    DEMO ENDPOINT — AI-assisted regulatory document drafting.

    Example:
      POST /api/v1/pharma/draft
      {
        "document_type": "csr",
        "drug_name": "dabrafenib",
        "indication": "BRAF V600E metastatic melanoma",
        "nct_id": "NCT01227889"
      }

    Returns a structured ICH E3-compliant CSR draft grounded in live
    ClinicalTrials.gov data and PubMed literature.
    """
    return await draft_regulatory_document(
        document_type=request.document_type,
        drug_name=request.drug_name,
        indication=request.indication,
        nct_id=request.nct_id,
        study_data=request.study_data,
        section=request.section,
    )


@router.get("/document-types")
async def get_document_types():
    """List supported regulatory document types."""
    return {
        doc_type: {
            "title": tmpl["title"],
            "guideline": tmpl["guideline"],
            "fda_reference": tmpl["fda_ref"],
        }
        for doc_type, tmpl in DOCUMENT_TEMPLATES.items()
    }


@router.get("/health")
async def health():
    return {"status": "ok", "layer": "ariston_pharma"}
