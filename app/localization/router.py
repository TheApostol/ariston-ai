"""
LatAm Localization — FastAPI Router.

Endpoints:
  POST /localization/detect         — detect language of text
  POST /localization/translate      — translate text to target locale
  POST /localization/translate/batch — translate to multiple locales
  GET  /localization/regulatory/map — map FDA requirement → LatAm agency
  GET  /localization/regulatory/all — all mappings for a locale
  GET  /localization/partners       — query LatAm partner database
  GET  /localization/partners/{id}  — get specific partner
  GET  /localization/config/{locale} — locale-specific configuration
  GET  /localization/locales        — list supported locales
  GET  /localization/health         — health check
"""

from fastapi import APIRouter, Query
from pydantic import BaseModel
from typing import List, Optional

from .service import detect_language, translate_text, batch_translate, SUPPORTED_LOCALES
from .regulatory_mapping import (
    map_requirement,
    get_all_mappings_for_locale,
    get_locale_config,
    get_agency_for_locale,
    REGULATORY_CROSSWALK,
)
from .partner_db import get_partners, get_partner_by_id, list_countries, list_specialties

router = APIRouter(prefix="/localization", tags=["LatAm Localization"])


# ── Request / Response models ─────────────────────────────────────────────────

class DetectRequest(BaseModel):
    text: str


class TranslateRequest(BaseModel):
    text: str
    target_locale: str           # "es-MX" | "es-CO" | "es-AR" | "pt-BR"
    source_locale: Optional[str] = None


class BatchTranslateRequest(BaseModel):
    text: str
    target_locales: List[str]


# ── Language detection ────────────────────────────────────────────────────────

@router.post("/detect")
async def detect(request: DetectRequest):
    """Detect the language of the provided text."""
    detected = detect_language(request.text)
    return {
        "detected_locale": detected,
        "character_count": len(request.text),
    }


# ── Translation ───────────────────────────────────────────────────────────────

@router.post("/translate")
async def translate(request: TranslateRequest):
    """
    Translate pharma/regulatory text to the target locale.

    Supported locales: es-MX, es-CO, es-AR, pt-BR

    Example:
      POST /api/v1/localization/translate
      { "text": "The patient experienced serious adverse events...", "target_locale": "pt-BR" }
    """
    return await translate_text(
        text=request.text,
        target_locale=request.target_locale,
        source_locale=request.source_locale,
    )


@router.post("/translate/batch")
async def translate_batch(request: BatchTranslateRequest):
    """
    Translate a single text to multiple locales simultaneously.

    Example:
      POST /api/v1/localization/translate/batch
      { "text": "...", "target_locales": ["es-MX", "pt-BR", "es-CO"] }
    """
    return await batch_translate(
        text=request.text,
        target_locales=request.target_locales,
    )


# ── Regulatory mapping ────────────────────────────────────────────────────────

@router.get("/regulatory/map")
async def regulatory_map(
    fda_requirement: str = Query(..., description="FDA requirement (e.g. NDA, IND, BLA)"),
    target_agency: str = Query(..., description="Target LatAm agency (ANVISA, COFEPRIS, INVIMA, ANMAT)"),
):
    """
    Map a specific FDA requirement to the equivalent LatAm agency pathway.

    Example:
      GET /api/v1/localization/regulatory/map?fda_requirement=NDA&target_agency=ANVISA
    """
    mapping = map_requirement(fda_requirement, target_agency)
    if not mapping:
        return {
            "fda_requirement": fda_requirement,
            "target_agency": target_agency,
            "error": f"No mapping found for '{fda_requirement}' → {target_agency}",
            "available_requirements": list(REGULATORY_CROSSWALK.keys()),
        }
    return {
        "fda_requirement": fda_requirement,
        "target_agency": target_agency,
        **mapping,
    }


@router.get("/regulatory/all")
async def regulatory_all(
    locale: str = Query(..., description="Target locale (e.g. pt-BR, es-MX, es-CO, es-AR)"),
):
    """
    Return all FDA → agency mappings for a given locale.

    Example:
      GET /api/v1/localization/regulatory/all?locale=pt-BR
    """
    agency = get_agency_for_locale(locale)
    mappings = get_all_mappings_for_locale(locale)
    return {
        "locale": locale,
        "agency": agency,
        "mappings": mappings,
        "total": len(mappings),
    }


@router.get("/regulatory/requirements")
async def list_requirements():
    """List all FDA requirements available for mapping."""
    return {
        "requirements": list(REGULATORY_CROSSWALK.keys()),
        "agencies": ["ANVISA", "COFEPRIS", "INVIMA", "ANMAT"],
    }


# ── Partner database ──────────────────────────────────────────────────────────

@router.get("/partners")
async def query_partners(
    country: Optional[str] = Query(None, description="Filter by country (Brazil, Mexico, Colombia, Argentina)"),
    locale: Optional[str] = Query(None, description="Filter by locale (pt-BR, es-MX, es-CO, es-AR)"),
    agency: Optional[str] = Query(None, description="Filter by agency (ANVISA, COFEPRIS, INVIMA, ANMAT)"),
    partner_type: Optional[str] = Query(None, description="Filter by type (cro, academic_hospital, regulatory_consultant)"),
    specialty: Optional[str] = Query(None, description="Filter by specialty (oncology, rare_disease, etc.)"),
):
    """
    Query the LatAm partner database.

    Example:
      GET /api/v1/localization/partners?country=Brazil&specialty=oncology
    """
    partners = get_partners(
        country=country,
        locale=locale,
        agency=agency,
        partner_type=partner_type,
        specialty=specialty,
    )
    return {
        "partners": partners,
        "total": len(partners),
        "filters_applied": {
            k: v for k, v in {
                "country": country, "locale": locale, "agency": agency,
                "type": partner_type, "specialty": specialty,
            }.items() if v
        },
    }


@router.get("/partners/meta")
async def partner_metadata():
    """Return available filter options for the partner database."""
    return {
        "countries": list_countries(),
        "specialties": list_specialties(),
        "partner_types": ["cro", "academic_hospital", "government_cancer_center",
                          "regulatory_consultant", "diagnostic_lab", "pharma_company",
                          "regulatory_network", "public_hospital"],
        "agencies": ["ANVISA", "COFEPRIS", "INVIMA", "ANMAT", "Multi-agency"],
    }


@router.get("/partners/{partner_id}")
async def get_partner(partner_id: str):
    """Retrieve a specific partner by ID."""
    partner = get_partner_by_id(partner_id)
    if not partner:
        return {"error": f"Partner '{partner_id}' not found"}
    return partner


# ── Locale configuration ──────────────────────────────────────────────────────

@router.get("/config/{locale}")
async def locale_config(locale: str):
    """
    Return locale-specific configuration (currency, units, date formats).

    Example:
      GET /api/v1/localization/config/pt-BR
    """
    config = get_locale_config(locale)
    agency = get_agency_for_locale(locale)
    return {
        "locale": locale,
        "locale_name": SUPPORTED_LOCALES.get(locale, "Unknown"),
        "agency": agency,
        "config": config,
    }


@router.get("/locales")
async def list_locales():
    """List all supported locales with their names and associated agencies."""
    return {
        "locales": [
            {
                "locale": locale,
                "name": name,
                "agency": get_agency_for_locale(locale),
                "config": get_locale_config(locale),
            }
            for locale, name in SUPPORTED_LOCALES.items()
        ]
    }


@router.get("/health")
async def health():
    return {"status": "ok", "layer": "localization", "supported_locales": list(SUPPORTED_LOCALES.keys())}
