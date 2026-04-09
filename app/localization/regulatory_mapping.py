"""
Regional regulatory mapping service.

Maps FDA requirements to LatAm agency equivalents:
  ANVISA (Brazil), COFEPRIS (Mexico), INVIMA (Colombia), ANMAT (Argentina)
"""

from typing import Optional

# ── Locale → Agency mapping ───────────────────────────────────────────────────
LOCALE_AGENCY_MAP = {
    "pt-BR": "ANVISA",
    "es-MX": "COFEPRIS",
    "es-CO": "INVIMA",
    "es-AR": "ANMAT",
    "en-US": "FDA",
    "en":    "FDA",
    "es":    "COFEPRIS",   # default Spanish → Mexico
    "pt":    "ANVISA",     # default Portuguese → Brazil
}

AGENCY_LOCALES = {
    "ANVISA":   "pt-BR",
    "COFEPRIS": "es-MX",
    "INVIMA":   "es-CO",
    "ANMAT":    "es-AR",
    "FDA":      "en-US",
}

# ── FDA → LatAm regulatory pathway mapping ───────────────────────────────────
# Structure: fda_requirement → {agency: {equivalent, reference, notes}}
REGULATORY_CROSSWALK = {
    # New Drug Application
    "NDA": {
        "ANVISA": {
            "equivalent": "Registro de Medicamento Novo (RMN)",
            "reference": "RDC 204/2017",
            "pathway": "Registro ordinário ou prioritário",
            "timeline_days": 365,
            "notes": "ANVISA prioritário pathway available for unmet needs — ~180 days",
        },
        "COFEPRIS": {
            "equivalent": "Registro Sanitario de Medicamento Nuevo",
            "reference": "Reglamento de Insumos para la Salud (RIS) Art. 167",
            "pathway": "Registro ordinario o acelerado",
            "timeline_days": 365,
            "notes": "COFEPRIS fast-track for orphan drugs or unmet medical need",
        },
        "INVIMA": {
            "equivalent": "Registro Sanitario de Medicamento Nuevo",
            "reference": "Decreto 677/1995 y Resolución 2048/2021",
            "pathway": "Registro ordinario o prioritario",
            "timeline_days": 730,
            "notes": "Can reference FDA approval for expedited review",
        },
        "ANMAT": {
            "equivalent": "Autorización de Comercialización de Especialidad Medicinal",
            "reference": "Disposición ANMAT 3311/10",
            "pathway": "Aprobación basada en agencia de referencia",
            "timeline_days": 365,
            "notes": "Can use FDA/EMA approval as reference for abridged review",
        },
    },

    # IND Application
    "IND": {
        "ANVISA": {
            "equivalent": "Autorização de Pesquisa Clínica (APC) / ANVISA",
            "reference": "RDC 9/2015 e Resolução 251/97",
            "pathway": "Submissão ao CEP + CONEP + ANVISA",
            "timeline_days": 90,
            "notes": "CEP (local IRB) + CONEP (national) approval required alongside ANVISA",
        },
        "COFEPRIS": {
            "equivalent": "Autorización de Protocolo de Investigación Clínica",
            "reference": "NOM-012-SSA3-2012",
            "pathway": "COFEPRIS + Comité de Ética + CONBIOÉTICA",
            "timeline_days": 60,
            "notes": "COFEPRIS clinical trial authorization + ethics committee approval",
        },
        "INVIMA": {
            "equivalent": "Permiso de Investigación Clínica",
            "reference": "Resolución 2378/2008 (BPC)",
            "pathway": "INVIMA + Comité de Ética Institucional",
            "timeline_days": 30,
            "notes": "Relatively streamlined for Phase I/II",
        },
        "ANMAT": {
            "equivalent": "Autorización de Ensayo Clínico",
            "reference": "Disposición ANMAT 6677/10",
            "pathway": "ANMAT + Comité de Ética + CONEAU",
            "timeline_days": 60,
            "notes": "Phase I requires ANMAT authorization; Phases II-IV with ethics approval",
        },
    },

    # Biologics License Application
    "BLA": {
        "ANVISA": {
            "equivalent": "Registro de Medicamento Biológico",
            "reference": "RDC 55/2010 (biológicos) / RDC 204/2017",
            "pathway": "Registro de Produto Biológico (RPB) ou PBP",
            "timeline_days": 540,
            "notes": "Biosimilar pathway under RDC 55/2010 with comparability exercise",
        },
        "COFEPRIS": {
            "equivalent": "Registro Sanitario de Biotecnológico o Biológico",
            "reference": "NOM-257-SSA1-2014 (bioterapéuticos) / RIS",
            "pathway": "Registro de biotecnológico innovador o biosimilar",
            "timeline_days": 540,
            "notes": "Biocomparability studies required for biosimilars",
        },
        "INVIMA": {
            "equivalent": "Registro Sanitario de Biológico",
            "reference": "Resolución 3622/2010 y Decreto 1782/2014",
            "pathway": "Medicamento biológico o biosimilar",
            "timeline_days": 730,
            "notes": "Biosimilar regulations Decreto 1782/2014",
        },
        "ANMAT": {
            "equivalent": "Autorización de Medicamento Biotecnológico",
            "reference": "Disposición ANMAT 7729/2011",
            "pathway": "Medicamento biológico o biosimilar",
            "timeline_days": 365,
            "notes": "Comparability protocol required; reference product concept per Disp. 7729/2011",
        },
    },

    # Pharmacovigilance reporting
    "MedWatch (FDA 3500A)": {
        "ANVISA": {
            "equivalent": "Notificação NOTIVISA — Vigilância Sanitária",
            "reference": "RDC 204/2017 / RDC 9/2015 / NOM PV",
            "pathway": "Portal NOTIVISA",
            "timeline_days": 15,
            "notes": "SAEs must be reported within 7 days (fatal/life-threatening) or 15 days (other)",
        },
        "COFEPRIS": {
            "equivalent": "Reporte VIGIFARMA — Sistema Nacional de Farmacovigilancia",
            "reference": "NOM-220-SSA1-2016",
            "pathway": "Portal VIGIFARMA",
            "timeline_days": 15,
            "notes": "Serious unexpected ADRs: 15-day expedited report; annual safety reports",
        },
        "INVIMA": {
            "equivalent": "Reporte al SIVIGILA / INVIMA",
            "reference": "Resolución 1403/2007 (PRM) y Decreto 677/95",
            "pathway": "Formulario INVIMA Farmacovigilancia",
            "timeline_days": 15,
            "notes": "Serious unexpected ADRs: 15 days; follow-up at 90 days",
        },
        "ANMAT": {
            "equivalent": "Notificación al Sistema Nacional de Farmacovigilancia (SNFVG)",
            "reference": "Disposición ANMAT 5358/12",
            "pathway": "Portal ANMAT Farmacovigilancia",
            "timeline_days": 15,
            "notes": "Deaths/life-threatening: 7 calendar days; other SAEs: 15 days",
        },
    },

    # Good Manufacturing Practice
    "GMP (21 CFR Parts 210-211)": {
        "ANVISA": {
            "equivalent": "Boas Práticas de Fabricação (BPF)",
            "reference": "RDC 301/2019 (BPF Medicamentos)",
            "pathway": "Certificado de Boas Práticas de Fabricação (CBPF)",
            "timeline_days": 180,
            "notes": "ICH Q7/Q10 aligned; CBPF mandatory for registration",
        },
        "COFEPRIS": {
            "equivalent": "Buenas Prácticas de Fabricación (BPF)",
            "reference": "NOM-059-SSA1-2015",
            "pathway": "Certificado de BPF emitido por COFEPRIS",
            "timeline_days": 180,
            "notes": "WHO GMP or PIC/S certificate accepted for imported products",
        },
        "INVIMA": {
            "equivalent": "Buenas Prácticas de Manufactura (BPM)",
            "reference": "Decreto 549/2001 y Resolución 1160/2016",
            "pathway": "Certificado de Cumplimiento BPM",
            "timeline_days": 90,
            "notes": "Accepts FDA, EMA, or PIC/S certificates for expedited review",
        },
        "ANMAT": {
            "equivalent": "Buenas Prácticas de Fabricación (BPF)",
            "reference": "Disposición ANMAT 2819/04 y Disposición 3827/18",
            "pathway": "Certificado de BPF (ANMAT o autoridad extranjera reconocida)",
            "timeline_days": 90,
            "notes": "FDA, EMA, PIC/S GMP certificates accepted as equivalent",
        },
    },

    # Orphan Drug Designation
    "Orphan Drug Designation (ODD)": {
        "ANVISA": {
            "equivalent": "Medicamento para Doença Rara ou Negligenciada",
            "reference": "Lei 13.689/2018 / Portaria Interministerial 11/2020",
            "pathway": "CONITEC avaliação + vias de acesso especial",
            "timeline_days": 180,
            "notes": "Access mechanisms: acesso expandido, uso compassivo, uso emergencial",
        },
        "COFEPRIS": {
            "equivalent": "Medicamento Huérfano",
            "reference": "Ley General de Salud Art. 224A",
            "pathway": "Registro prioritario de medicamento huérfano",
            "timeline_days": 180,
            "notes": "Expedited 6-month review; can reference FDA/EMA orphan approval",
        },
        "INVIMA": {
            "equivalent": "Medicamento Huérfano",
            "reference": "Ley 1392/2010 (Enfermedades Huérfanas) / Decreto 1954/2012",
            "pathway": "Registro prioritario",
            "timeline_days": 180,
            "notes": "Referencia a aprobación FDA/EMA para revisión abreviada",
        },
        "ANMAT": {
            "equivalent": "Medicamento Huérfano",
            "reference": "Ley 26.689 (Enfermedades Poco Frecuentes)",
            "pathway": "Registro de medicamento huérfano con vía acelerada",
            "timeline_days": 180,
            "notes": "Temporary access possible pending full registration",
        },
    },
}

# ── Drug naming conventions by region ────────────────────────────────────────
# INN (International Nonproprietary Name) equivalents by region
DRUG_NAMING_CONVENTIONS = {
    "FDA":      {"system": "USAN",   "label": "USAN (United States Adopted Name)"},
    "ANVISA":   {"system": "DCB",    "label": "DCB (Denominação Comum Brasileira)"},
    "COFEPRIS": {"system": "DCI",    "label": "DCI (Denominación Común Internacional)"},
    "INVIMA":   {"system": "DCI",    "label": "DCI (Denominación Común Internacional)"},
    "ANMAT":    {"system": "DCI",    "label": "DCI (Denominación Común Internacional)"},
    "EMA":      {"system": "INN",    "label": "INN (International Nonproprietary Name)"},
}

# ── Measurement & currency preferences by locale ─────────────────────────────
LOCALE_PREFERENCES = {
    "pt-BR": {
        "currency": "BRL",
        "currency_symbol": "R$",
        "date_format": "DD/MM/YYYY",
        "decimal_separator": ",",
        "thousand_separator": ".",
        "weight_unit": "kg",
        "volume_unit": "mL",
        "temperature_unit": "°C",
    },
    "es-MX": {
        "currency": "MXN",
        "currency_symbol": "$",
        "date_format": "DD/MM/YYYY",
        "decimal_separator": ".",
        "thousand_separator": ",",
        "weight_unit": "kg",
        "volume_unit": "mL",
        "temperature_unit": "°C",
    },
    "es-CO": {
        "currency": "COP",
        "currency_symbol": "$",
        "date_format": "DD/MM/YYYY",
        "decimal_separator": ",",
        "thousand_separator": ".",
        "weight_unit": "kg",
        "volume_unit": "mL",
        "temperature_unit": "°C",
    },
    "es-AR": {
        "currency": "ARS",
        "currency_symbol": "$",
        "date_format": "DD/MM/YYYY",
        "decimal_separator": ",",
        "thousand_separator": ".",
        "weight_unit": "kg",
        "volume_unit": "mL",
        "temperature_unit": "°C",
    },
    "en-US": {
        "currency": "USD",
        "currency_symbol": "$",
        "date_format": "MM/DD/YYYY",
        "decimal_separator": ".",
        "thousand_separator": ",",
        "weight_unit": "kg",
        "volume_unit": "mL",
        "temperature_unit": "°C",
    },
}


def get_agency_for_locale(locale: str) -> str:
    """Return the primary regulatory agency for a given locale."""
    return LOCALE_AGENCY_MAP.get(locale, "FDA")


def get_locale_for_agency(agency: str) -> str:
    """Return the primary locale for a given agency."""
    return AGENCY_LOCALES.get(agency.upper(), "en-US")


def map_requirement(fda_requirement: str, target_agency: str) -> Optional[dict]:
    """
    Map an FDA regulatory requirement to the target LatAm agency equivalent.

    Args:
        fda_requirement: e.g. "NDA", "IND", "BLA", "MedWatch (FDA 3500A)"
        target_agency: e.g. "ANVISA", "COFEPRIS", "INVIMA", "ANMAT"

    Returns:
        dict with equivalent, reference, pathway, timeline_days, notes
        or None if mapping not found
    """
    crosswalk = REGULATORY_CROSSWALK.get(fda_requirement, {})
    return crosswalk.get(target_agency.upper())


def get_all_mappings_for_locale(locale: str) -> dict:
    """Return all FDA → agency mappings for a given locale."""
    agency = get_agency_for_locale(locale)
    result = {}
    for fda_req, agency_map in REGULATORY_CROSSWALK.items():
        if agency in agency_map:
            result[fda_req] = agency_map[agency]
    return result


def get_locale_config(locale: str) -> dict:
    """Return locale-specific configuration (currency, units, formats)."""
    return LOCALE_PREFERENCES.get(locale, LOCALE_PREFERENCES["en-US"])
