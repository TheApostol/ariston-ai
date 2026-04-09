"""
LatAm CRO, health system, and biotech partner database.

Lightweight in-memory database of relevant LatAm organizations
for pharma/biotech pilots.
"""

from typing import List, Optional

# ── Partner records ───────────────────────────────────────────────────────────
PARTNER_DATABASE: List[dict] = [
    # ── Brazil (ANVISA) ──────────────────────────────────────────────────────
    {
        "id": "br-001",
        "name": "Hospital das Clínicas da FMUSP",
        "country": "Brazil",
        "locale": "pt-BR",
        "agency": "ANVISA",
        "type": "academic_hospital",
        "specialties": ["oncology", "cardiology", "rare_disease", "neurology"],
        "city": "São Paulo",
        "contact_email": "pesquisaclinica@hc.fm.usp.br",
        "website": "https://www.hc.fm.usp.br",
        "notes": "Largest academic medical center in Latin America; strong Phase I-III capability",
    },
    {
        "id": "br-002",
        "name": "Instituto Nacional de Câncer (INCA)",
        "country": "Brazil",
        "locale": "pt-BR",
        "agency": "ANVISA",
        "type": "government_cancer_center",
        "specialties": ["oncology", "hematology"],
        "city": "Rio de Janeiro",
        "contact_email": "ensaios@inca.gov.br",
        "website": "https://www.inca.gov.br",
        "notes": "National cancer reference center; oncology trial expertise",
    },
    {
        "id": "br-003",
        "name": "ICON Brazil (São Paulo)",
        "country": "Brazil",
        "locale": "pt-BR",
        "agency": "ANVISA",
        "type": "cro",
        "specialties": ["oncology", "cardiology", "rare_disease", "infectious_disease"],
        "city": "São Paulo",
        "contact_email": "brazil@iconplc.com",
        "website": "https://www.iconplc.com",
        "notes": "Full-service CRO with ANVISA submission expertise",
    },
    {
        "id": "br-004",
        "name": "Synapse — Consultoria Regulatória",
        "country": "Brazil",
        "locale": "pt-BR",
        "agency": "ANVISA",
        "type": "regulatory_consultant",
        "specialties": ["anvisa_registration", "clinical_trials", "pharmacovigilance"],
        "city": "São Paulo",
        "contact_email": "contato@synapsebr.com.br",
        "website": "https://www.synapsebr.com.br",
        "notes": "ANVISA regulatory strategy and dossier preparation specialist",
    },
    {
        "id": "br-005",
        "name": "Grupo Fleury",
        "country": "Brazil",
        "locale": "pt-BR",
        "agency": "ANVISA",
        "type": "diagnostic_lab",
        "specialties": ["diagnostics", "genomics", "biomarkers"],
        "city": "São Paulo",
        "contact_email": "negocios@grupofleury.com.br",
        "website": "https://www.grupofleury.com.br",
        "notes": "Largest private diagnostic lab network in Brazil; biomarker data",
    },

    # ── Mexico (COFEPRIS) ────────────────────────────────────────────────────
    {
        "id": "mx-001",
        "name": "Instituto Nacional de Ciencias Médicas y Nutrición Salvador Zubirán (INCMNSZ)",
        "country": "Mexico",
        "locale": "es-MX",
        "agency": "COFEPRIS",
        "type": "academic_hospital",
        "specialties": ["oncology", "metabolic", "rare_disease", "gastroenterology"],
        "city": "Mexico City",
        "contact_email": "investigacion.clinica@innsz.mx",
        "website": "https://www.innsz.mx",
        "notes": "Top-ranked clinical research center in Mexico; COFEPRIS trial expertise",
    },
    {
        "id": "mx-002",
        "name": "Intramed — Investigación Clínica",
        "country": "Mexico",
        "locale": "es-MX",
        "agency": "COFEPRIS",
        "type": "cro",
        "specialties": ["oncology", "cardiology", "infectious_disease", "neurology"],
        "city": "Monterrey",
        "contact_email": "negocios@intramed.com.mx",
        "website": "https://www.intramed.com.mx",
        "notes": "Full-service CRO with NOM-012 compliance expertise",
    },
    {
        "id": "mx-003",
        "name": "Hospital General de México Dr. Eduardo Liceaga",
        "country": "Mexico",
        "locale": "es-MX",
        "agency": "COFEPRIS",
        "type": "public_hospital",
        "specialties": ["oncology", "cardiology", "infectious_disease"],
        "city": "Mexico City",
        "contact_email": "investigacion@hgm.salud.gob.mx",
        "website": "https://www.hgm.salud.gob.mx",
        "notes": "Large patient population; government reference hospital",
    },
    {
        "id": "mx-004",
        "name": "Iqvia México",
        "country": "Mexico",
        "locale": "es-MX",
        "agency": "COFEPRIS",
        "type": "cro",
        "specialties": ["data_analytics", "rwe", "pharmacovigilance", "regulatory"],
        "city": "Mexico City",
        "contact_email": "mexico@iqvia.com",
        "website": "https://www.iqvia.com/locations/latin-america/mexico",
        "notes": "Global CRO with strong LatAm data analytics and RWE capabilities",
    },

    # ── Colombia (INVIMA) ────────────────────────────────────────────────────
    {
        "id": "co-001",
        "name": "Fundación Santa Fe de Bogotá",
        "country": "Colombia",
        "locale": "es-CO",
        "agency": "INVIMA",
        "type": "academic_hospital",
        "specialties": ["oncology", "cardiology", "infectious_disease", "rare_disease"],
        "city": "Bogotá",
        "contact_email": "investigacion@fsfb.org.co",
        "website": "https://www.fsfb.org.co",
        "notes": "Leading academic hospital; BPC-compliant trials; INVIMA submission experience",
    },
    {
        "id": "co-002",
        "name": "ClinicalResearch.lat (Colombia)",
        "country": "Colombia",
        "locale": "es-CO",
        "agency": "INVIMA",
        "type": "cro",
        "specialties": ["clinical_trials", "regulatory", "pharmacovigilance"],
        "city": "Bogotá",
        "contact_email": "info@clinicalresearch.lat",
        "website": "https://clinicalresearch.lat",
        "notes": "LatAm-focused CRO; strong INVIMA regulatory pathway knowledge",
    },
    {
        "id": "co-003",
        "name": "Instituto Nacional de Cancerología (INC Colombia)",
        "country": "Colombia",
        "locale": "es-CO",
        "agency": "INVIMA",
        "type": "government_cancer_center",
        "specialties": ["oncology", "hematology"],
        "city": "Bogotá",
        "contact_email": "investigacion@cancer.gov.co",
        "website": "https://www.cancer.gov.co",
        "notes": "National oncology reference; Colombian cancer registry access",
    },

    # ── Argentina (ANMAT) ────────────────────────────────────────────────────
    {
        "id": "ar-001",
        "name": "Hospital Italiano de Buenos Aires",
        "country": "Argentina",
        "locale": "es-AR",
        "agency": "ANMAT",
        "type": "academic_hospital",
        "specialties": ["oncology", "cardiology", "rare_disease", "genomics"],
        "city": "Buenos Aires",
        "contact_email": "investigacion.clinica@hospitalitaliano.org.ar",
        "website": "https://www.hospitalitaliano.org.ar",
        "notes": "Top private academic hospital; EHR system with research data access",
    },
    {
        "id": "ar-002",
        "name": "Fundación CEMIC",
        "country": "Argentina",
        "locale": "es-AR",
        "agency": "ANMAT",
        "type": "academic_hospital",
        "specialties": ["cardiology", "oncology", "endocrinology"],
        "city": "Buenos Aires",
        "contact_email": "investigacion@cemic.edu.ar",
        "website": "https://www.cemic.edu.ar",
        "notes": "Academic research center; ANMAT GCP-compliant trial history",
    },
    {
        "id": "ar-003",
        "name": "Laboratorio Roemmers (Argentina)",
        "country": "Argentina",
        "locale": "es-AR",
        "agency": "ANMAT",
        "type": "pharma_company",
        "specialties": ["pharmaceutical_manufacturing", "regulatory", "distribution"],
        "city": "Buenos Aires",
        "contact_email": "roemmers@roemmers.com.ar",
        "website": "https://www.roemmers.com.ar",
        "notes": "Largest Argentine pharma company; potential distribution partner",
    },

    # ── Pan-LatAm CROs ───────────────────────────────────────────────────────
    {
        "id": "latam-001",
        "name": "Parexel LatAm",
        "country": "Multi-country",
        "locale": "es",
        "agency": "Multi-agency",
        "type": "cro",
        "specialties": ["clinical_trials", "regulatory", "data_management", "pharmacovigilance"],
        "city": "Buenos Aires / São Paulo / Mexico City",
        "contact_email": "latinamerica@parexel.com",
        "website": "https://www.parexel.com/regions/latin-america",
        "notes": "Pan-LatAm CRO with ANVISA, COFEPRIS, INVIMA, ANMAT submission experience",
    },
    {
        "id": "latam-002",
        "name": "Syneos Health LatAm",
        "country": "Multi-country",
        "locale": "es",
        "agency": "Multi-agency",
        "type": "cro",
        "specialties": ["clinical_trials", "regulatory", "rwe"],
        "city": "São Paulo / Mexico City",
        "contact_email": "latinamerica@syneoshealth.com",
        "website": "https://www.syneoshealth.com",
        "notes": "Full-service biopharmaceutical services; LatAm region expertise",
    },
    {
        "id": "latam-003",
        "name": "LATAM Health Regulatory Network (LHRN)",
        "country": "Multi-country",
        "locale": "es",
        "agency": "Multi-agency",
        "type": "regulatory_network",
        "specialties": ["regulatory_strategy", "dossier_preparation", "government_affairs"],
        "city": "Multiple cities",
        "contact_email": "info@lhrn.lat",
        "website": "https://www.lhrn.lat",
        "notes": "Regional network of regulatory affairs specialists; ANVISA/COFEPRIS/INVIMA/ANMAT",
    },
]

# ── Country code → locale map for convenience ─────────────────────────────────
COUNTRY_LOCALE_MAP = {
    "Brazil":       "pt-BR",
    "Mexico":       "es-MX",
    "Colombia":     "es-CO",
    "Argentina":    "es-AR",
    "Multi-country": "es",
}


def get_partners(
    country: Optional[str] = None,
    locale: Optional[str] = None,
    agency: Optional[str] = None,
    partner_type: Optional[str] = None,
    specialty: Optional[str] = None,
) -> List[dict]:
    """
    Query the partner database with optional filters.

    Args:
        country: e.g. "Brazil", "Mexico", "Colombia", "Argentina"
        locale: e.g. "pt-BR", "es-MX", "es-CO", "es-AR"
        agency: e.g. "ANVISA", "COFEPRIS", "INVIMA", "ANMAT"
        partner_type: e.g. "cro", "academic_hospital", "regulatory_consultant"
        specialty: e.g. "oncology", "rare_disease", "pharmacovigilance"

    Returns:
        Filtered list of partner records.
    """
    results = PARTNER_DATABASE

    if country:
        results = [p for p in results if country.lower() in p["country"].lower()]

    if locale:
        results = [p for p in results if locale.lower() in p["locale"].lower()]

    if agency:
        results = [
            p for p in results
            if agency.upper() in p["agency"].upper()
        ]

    if partner_type:
        results = [p for p in results if partner_type.lower() in p["type"].lower()]

    if specialty:
        results = [
            p for p in results
            if any(specialty.lower() in s.lower() for s in p["specialties"])
        ]

    return results


def get_partner_by_id(partner_id: str) -> Optional[dict]:
    """Return a single partner by ID."""
    for partner in PARTNER_DATABASE:
        if partner["id"] == partner_id:
            return partner
    return None


def list_countries() -> List[str]:
    """List all countries represented in the database."""
    return sorted({p["country"] for p in PARTNER_DATABASE})


def list_specialties() -> List[str]:
    """List all specialties across all partners."""
    specialties = set()
    for p in PARTNER_DATABASE:
        specialties.update(p["specialties"])
    return sorted(specialties)
