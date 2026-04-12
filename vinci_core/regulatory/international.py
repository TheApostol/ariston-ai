"""
International Regulatory Intelligence — Phase 3 / Ariston AI.

Extends beyond LATAM to cover major global regulatory authorities:
  - EMA   (European Medicines Agency) — EU/EEA
  - PMDA  (Pharmaceuticals and Medical Devices Agency) — Japan
  - ANVS  (Agência Nacional de Vigilância Sanitária — EEA scope) — note: ANVISA is Brazil
  - MHRA  (Medicines and Healthcare products Regulatory Agency) — UK post-Brexit
  - TGA   (Therapeutic Goods Administration) — Australia
  - COFEPRIS / ANVISA in scope from latam_layer.py (not repeated here)

This module provides:
  1. Regulatory authority registry (submission pathways, timelines, fees)
  2. Dossier gap analysis (what LATAM approval needs for EMA/PMDA recognition)
  3. International harmonization map (ICH guidelines applicability)
  4. Parallel submission strategy (file in multiple regions simultaneously)

Strategic value: Phase 3 international expansion — once LATAM revenue validates
the model, expand to EU + Japan where regulatory intelligence SaaS commands
higher willingness-to-pay ($500K–$2M/year for enterprise contracts).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("ariston.regulatory.international")

# ---------------------------------------------------------------------------
# Authority Registry
# ---------------------------------------------------------------------------
INTERNATIONAL_AUTHORITIES: dict[str, dict] = {
    "ema": {
        "name": "European Medicines Agency",
        "region": "EU/EEA (27 member states + Iceland, Liechtenstein, Norway)",
        "headquarters": "Amsterdam, Netherlands",
        "submission_pathways": {
            "centralized": {
                "description": "Single application → valid in all EU/EEA member states",
                "timeline_months": "12–15",
                "mandatory_for": ["biotech/biologics", "oncology", "rare diseases", "advanced therapies"],
                "fee_eur": "322,200",
            },
            "decentralized": {
                "description": "Simultaneous submission to multiple member states",
                "timeline_months": "15–18",
                "mandatory_for": ["products not requiring centralized procedure"],
                "fee_eur": "Varies by country",
            },
            "mutual_recognition": {
                "description": "Extension of existing national authorization to other member states",
                "timeline_months": "6–9",
                "mandatory_for": ["already approved in one EU member state"],
                "fee_eur": "Varies",
            },
        },
        "key_guidelines": ["ICH E6(R2) GCP", "ICH E3 CSR", "ICH E2B(R3) PhV", "EMA IMPD guidance"],
        "latam_recognition": "No direct recognition — full EMA dossier required; CTD format (ICH M4) accepted",
        "ai_device_pathway": "EU MDR 2017/745 for SaMD; AI/ML guidance under CE marking",
        "contact": "info@ema.europa.eu",
        "portal": "IRIS submission platform",
        "expedited_programs": ["PRIME (PRIority MEdicines)", "Accelerated Assessment", "Conditional MA"],
    },
    "pmda": {
        "name": "Pharmaceuticals and Medical Devices Agency",
        "region": "Japan",
        "headquarters": "Tokyo, Japan",
        "submission_pathways": {
            "new_drug_application": {
                "description": "Shinsho — new molecular entity approval",
                "timeline_months": "12",
                "mandatory_for": ["all new drugs entering Japan market"],
                "fee_jpy": "varies by priority designation",
            },
            "drug_master_file": {
                "description": "Master File registration for API/excipients",
                "timeline_months": "3–6",
                "mandatory_for": ["API manufacturers supplying Japan"],
            },
        },
        "key_guidelines": ["ICH E6(R2)", "ICH M4 CTD", "J-GCP (GCP Ordinance)", "AMED clinical trial guidance"],
        "latam_recognition": "No direct recognition; bridging studies often required for ethnic sensitivity (ICH E5)",
        "ai_device_pathway": "PMDA SaMD guidance 2021; IMDRF AI/ML framework adopted",
        "contact": "https://www.pmda.go.jp",
        "portal": "PMDA eSubmission Gateway",
        "expedited_programs": ["Sakigake Designation (breakthrough)", "Conditional early approval"],
        "ethnic_requirements": "ICH E5 bridging study often required; Japan-specific pharmacogenomics data",
    },
    "mhra": {
        "name": "Medicines and Healthcare products Regulatory Agency",
        "region": "United Kingdom (post-Brexit)",
        "headquarters": "London, UK",
        "submission_pathways": {
            "ukma": {
                "description": "UK Marketing Authorisation (post-Brexit standalone pathway)",
                "timeline_months": "12–15",
                "mandatory_for": ["all medicines sold in UK market"],
                "fee_gbp": "varies by application type",
            },
            "access_consortium": {
                "description": "Work-sharing with TGA (Australia), Health Canada, Swissmedic, MCC (Singapore)",
                "timeline_months": "12",
                "mandatory_for": ["parallel submission strategy"],
            },
        },
        "key_guidelines": ["UK equivalent of ICH guidelines", "MHRA SaMD guidance"],
        "latam_recognition": "No direct recognition; CTD format accepted",
        "ai_device_pathway": "MHRA AI/ML Software as a Medical Device roadmap 2023",
        "expedited_programs": ["Innovative Licensing and Access Pathway (ILAP)", "Rolling Review"],
    },
    "tga": {
        "name": "Therapeutic Goods Administration",
        "region": "Australia",
        "headquarters": "Canberra, Australia",
        "submission_pathways": {
            "aust_r": {
                "description": "AUST R registration for prescription medicines",
                "timeline_months": "12",
                "fee_aud": "varies",
            },
        },
        "key_guidelines": ["ICH guidelines adopted", "TGA SaMD guidance"],
        "latam_recognition": "Access Consortium work-sharing pathway with MHRA/Health Canada",
        "expedited_programs": ["Priority Review", "Provisional Determination"],
    },
    "health_canada": {
        "name": "Health Canada",
        "region": "Canada",
        "headquarters": "Ottawa, Canada",
        "submission_pathways": {
            "nds": {
                "description": "New Drug Submission",
                "timeline_months": "12",
            },
        },
        "key_guidelines": ["ICH guidelines adopted", "Canada Food and Drugs Act"],
        "latam_recognition": "Access Consortium work-sharing",
        "expedited_programs": ["Priority Review", "Notice of Compliance with Conditions"],
    },
}

# ICH guideline applicability matrix
ICH_GUIDELINES: dict[str, dict] = {
    "E3":   {"title": "Structure and Content of CSR",            "applies_to": ["ema", "pmda", "mhra", "tga", "health_canada", "fda"], "ariston_module": "csr_pipeline"},
    "E6R2": {"title": "Good Clinical Practice",                  "applies_to": ["ema", "pmda", "mhra", "tga", "health_canada", "fda"], "ariston_module": "clinical_trial_pipeline"},
    "E2B":  {"title": "Pharmacovigilance (ICSR transmission)",   "applies_to": ["ema", "pmda", "mhra", "tga", "health_canada", "fda"], "ariston_module": "pharmacovigilance_pipeline"},
    "E5":   {"title": "Ethnic Factors in Acceptance of Foreign Data", "applies_to": ["pmda", "health_canada"],                         "ariston_module": "clinical_trial_pipeline"},
    "M4":   {"title": "Common Technical Document (CTD) format",  "applies_to": ["ema", "pmda", "mhra", "tga", "health_canada", "fda"], "ariston_module": "csr_pipeline"},
    "M7":   {"title": "Mutagenic Impurities",                    "applies_to": ["ema", "pmda", "mhra", "fda"],                         "ariston_module": None},
    "S9":   {"title": "Nonclinical Evaluation for Anticancer",   "applies_to": ["ema", "pmda", "fda"],                                 "ariston_module": None},
}


@dataclass
class InternationalDossierGap:
    """Gap between an existing LATAM dossier and a target authority's requirements."""
    authority: str
    authority_name: str
    region: str
    gaps: list[str]
    required_studies: list[str]
    estimated_bridging_cost_usd: str
    estimated_timeline_months: str
    recommended_pathway: str
    parallel_submission_eligible: bool
    applicable_ich_guidelines: list[str]
    expedited_programs: list[str] = field(default_factory=list)


class InternationalRegulatoryEngine:
    """
    Regulatory intelligence for international market expansion beyond LATAM.
    """

    def analyze_expansion(
        self,
        product_type: str,
        existing_approvals: list[str],
        target_authorities: list[str],
        indication: str,
        is_samd: bool = False,
    ) -> list[InternationalDossierGap]:
        """
        Analyze gaps for international expansion from LATAM base.

        Args:
            product_type: small_molecule | biologic | SaMD | combination
            existing_approvals: list of already-approved authorities (e.g. ["anvisa", "cofepris"])
            target_authorities: authorities to expand into (e.g. ["ema", "pmda"])
            indication: therapeutic indication
            is_samd: True if Software as a Medical Device

        Returns:
            List of gap analyses per target authority
        """
        gaps = []
        for authority_key in target_authorities:
            auth = INTERNATIONAL_AUTHORITIES.get(authority_key.lower())
            if not auth:
                logger.warning("[International] Unknown authority: %s", authority_key)
                continue
            gap = self._build_gap(authority_key, auth, product_type, existing_approvals, indication, is_samd)
            gaps.append(gap)
        return gaps

    def get_parallel_submission_strategy(
        self,
        target_authorities: list[str],
        product_type: str,
    ) -> dict:
        """
        Build a parallel submission strategy to maximize time efficiency.
        Groups authorities by workflow compatibility (Access Consortium etc.)
        """
        access_consortium = {"mhra", "tga", "health_canada"}
        target_set = set(t.lower() for t in target_authorities)

        strategy = {
            "independent_submissions": [],
            "parallel_groups": [],
            "recommended_order": [],
            "estimated_total_timeline_months": 0,
        }

        # Group Access Consortium members
        consortium_targets = target_set & access_consortium
        if len(consortium_targets) > 1:
            strategy["parallel_groups"].append({
                "group": "Access Consortium work-sharing",
                "authorities": list(consortium_targets),
                "timeline_reduction_pct": 30,
            })

        # EMA and PMDA are always independent
        for auth in ["ema", "pmda"]:
            if auth in target_set:
                strategy["independent_submissions"].append(auth)

        # Recommended order: largest market first (EMA > PMDA > MHRA > TGA > Health Canada)
        order_priority = ["ema", "pmda", "mhra", "tga", "health_canada"]
        strategy["recommended_order"] = [a for a in order_priority if a in target_set]

        # Rough timeline: longest independent submission
        max_months = 0
        for auth_key in target_set:
            auth = INTERNATIONAL_AUTHORITIES.get(auth_key, {})
            pathways = auth.get("submission_pathways", {})
            if pathways:
                first_pathway = next(iter(pathways.values()))
                tl = first_pathway.get("timeline_months", "12")
                months = int(str(tl).split("–")[0])
                max_months = max(max_months, months)

        strategy["estimated_total_timeline_months"] = max_months
        return strategy

    def get_ich_applicability(self, authority: str) -> list[dict]:
        """Return ICH guidelines applicable to a given authority."""
        auth_lower = authority.lower()
        return [
            {
                "guideline": code,
                "title": info["title"],
                "ariston_module": info.get("ariston_module"),
            }
            for code, info in ICH_GUIDELINES.items()
            if auth_lower in info["applies_to"]
        ]

    # ---------------------------------------------------------------------------
    # Private
    # ---------------------------------------------------------------------------

    def _build_gap(
        self,
        authority_key: str,
        auth: dict,
        product_type: str,
        existing_approvals: list[str],
        indication: str,
        is_samd: bool,
    ) -> InternationalDossierGap:
        has_latam = any(a in ["anvisa", "cofepris", "invima", "anmat", "isp"] for a in existing_approvals)
        gaps = []
        required_studies = []
        cost = "500K–2M"
        timeline = "12–18"
        pathway = next(iter(auth.get("submission_pathways", {}).keys()), "standard")

        # Common gaps for any non-recognized dossier
        if not has_latam:
            gaps.append("No prior regulatory approval — full new dossier required")
        else:
            gaps.append("LATAM approval not directly recognized — full CTD submission required")
            gaps.append("ICH M4 CTD format assembly required")

        # Authority-specific gaps
        if authority_key == "ema":
            gaps += [
                "EMA IMPD (Investigational Medicinal Product Dossier) required",
                "Risk Management Plan (RMP) required per EU GVP Module V",
                "EPAR (European Public Assessment Report) process — public disclosure",
            ]
            required_studies += ["EU-specific labeling (SmPC)", "PSUR schedule alignment"]
            if is_samd:
                gaps.append("CE marking under EU MDR 2017/745 required for SaMD")
                required_studies.append("Clinical Evaluation Report (CER) per EU MDR Annex XIV")
            cost = "1M–3M"
            timeline = "12–15"

        elif authority_key == "pmda":
            gaps += [
                "ICH E5 bridging study often required (ethnic sensitivity data)",
                "J-GCP compliance review",
                "Japanese-language package insert required",
                "Japan-specific pharmacovigilance plan",
            ]
            required_studies += [
                "Ethnic bridging study (if no Japanese patients in pivotal trial)",
                "Japanese PK/PD substudy (if PK differs by ethnicity)",
            ]
            cost = "800K–2.5M"
            timeline = "12–18"

        elif authority_key == "mhra":
            gaps += [
                "UK-specific labeling (SmPC) post-Brexit",
                "UK Clinical Trial Authorisation (CTA) if new trials needed",
            ]
            cost = "400K–1.2M"
            timeline = "12–15"

        # Parallel submission eligibility
        parallel_eligible = authority_key in {"mhra", "tga", "health_canada"}

        # ICH guidelines
        applicable_ich = [
            code for code, info in ICH_GUIDELINES.items()
            if authority_key in info["applies_to"]
        ]

        expedited = auth.get("expedited_programs", [])

        return InternationalDossierGap(
            authority=authority_key,
            authority_name=auth["name"],
            region=auth["region"],
            gaps=gaps,
            required_studies=required_studies,
            estimated_bridging_cost_usd=cost,
            estimated_timeline_months=timeline,
            recommended_pathway=pathway,
            parallel_submission_eligible=parallel_eligible,
            applicable_ich_guidelines=applicable_ich,
            expedited_programs=expedited,
        )


international_regulatory_engine = InternationalRegulatoryEngine()
