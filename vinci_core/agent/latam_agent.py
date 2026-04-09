"""
LATAM Regulatory Agent — Ariston AI.

Handles regulatory intelligence for Latin American pharmaceutical markets.
Covers ANVISA, COFEPRIS, INVIMA, ANMAT, ISP and PANDRH harmonization.

Primary use case for LATAM go-to-market:
- Regulatory document drafting (clinical study reports, dossiers)
- Submission strategy per country
- Cross-country registration roadmaps
- Bioequivalence gap analysis
- Pharmacovigilance reporting (VIGIMED, FARMACOVIGILANCIA)
"""

from __future__ import annotations
from typing import Optional


LATAM_AGENCIES = {
    "brazil": {
        "agency": "ANVISA",
        "full_name": "Agência Nacional de Vigilância Sanitária",
        "language": "Portuguese",
        "system": "SOLICITA",
        "key_regulations": ["RDC 204/2017", "RDC 73/2016", "Lei 6.360/1976"],
        "review_days_range": (90, 365),
        "priority_pathway": "Doenças negligenciadas, produtos inovadores",
    },
    "mexico": {
        "agency": "COFEPRIS",
        "full_name": "Comisión Federal para la Protección contra Riesgos Sanitarios",
        "language": "Spanish",
        "system": "Ventanilla Digital COFEPRIS",
        "key_regulations": ["NOM-072-SSA1", "NOM-177-SSA1", "FEUM"],
        "review_days_range": (180, 540),
        "priority_pathway": "Productos con aprobación FDA/EMA (vía abreviada)",
    },
    "colombia": {
        "agency": "INVIMA",
        "full_name": "Instituto Nacional de Vigilancia de Medicamentos y Alimentos",
        "language": "Spanish",
        "system": "SIVICOS",
        "key_regulations": ["Decreto 677/1995", "Resolución 2004009455", "Decreto 1782/2014"],
        "review_days_range": (60, 180),
        "priority_pathway": "Lista de Medicamentos Esenciales (LME)",
    },
    "argentina": {
        "agency": "ANMAT",
        "full_name": "Administración Nacional de Medicamentos, Alimentos y Tecnología Médica",
        "language": "Spanish",
        "system": "SAID",
        "key_regulations": ["Decreto 150/92", "Disposición 5904/1996", "Disposición 3185/99"],
        "review_days_range": (180, 730),
        "priority_pathway": "Reconocimiento de FDA/EMA/Health Canada (Disp. 3185/99)",
    },
    "chile": {
        "agency": "ISP",
        "full_name": "Instituto de Salud Pública de Chile",
        "language": "Spanish",
        "system": "ISP Portal Digital",
        "key_regulations": ["DFL 1/2005", "DS 3/2010"],
        "review_days_range": (90, 365),
        "priority_pathway": "Reconocimiento MERCOSUR / FDA/EMA",
    },
}

HARMONIZATION_FRAMEWORKS = [
    "PANDRH (Pan American Network for Drug Regulatory Harmonization)",
    "MERCOSUR Resolutions GMC 49/02, 36/06 (pharmaceutical harmonization)",
    "ICH Q1A-Q1F (stability), Q2R1 (validation), Q8-Q11 (pharmaceutical development)",
    "WHO Prequalification Programme",
    "IAEA GRP (Good Regulatory Practice for LATAM)",
]


class LatamRegulatoryAgent:
    """
    Regulatory intelligence agent for LATAM pharmaceutical registration.
    Returns structured submission strategies and country-specific guidance.
    """

    def get_agency_profile(self, country: str) -> dict:
        """Return agency profile for a given country."""
        return LATAM_AGENCIES.get(country.lower(), {})

    def all_countries(self) -> list[str]:
        return list(LATAM_AGENCIES.keys())

    def build_multi_country_roadmap(
        self,
        countries: list[str],
        product_type: str = "pharmaceutical",
        has_fda_approval: bool = False,
        has_ema_approval: bool = False,
    ) -> dict:
        """
        Build a multi-country LATAM registration roadmap.

        Args:
            countries: list of target LATAM countries
            product_type: 'pharmaceutical', 'biologic', 'medical_device', 'diagnostic'
            has_fda_approval: whether FDA clearance/approval exists
            has_ema_approval: whether EMA approval exists

        Returns:
            Structured roadmap with per-country strategy and timelines
        """
        roadmap = {
            "product_type": product_type,
            "reference_approvals": {
                "fda": has_fda_approval,
                "ema": has_ema_approval,
            },
            "countries": {},
            "harmonization_leverage": HARMONIZATION_FRAMEWORKS,
            "recommended_sequence": [],
            "total_estimated_timeline_months": 0,
        }

        priority_order = []
        for country in countries:
            profile = self.get_agency_profile(country)
            if not profile:
                continue

            expedited = has_fda_approval or has_ema_approval
            min_days, max_days = profile["review_days_range"]
            if expedited:
                # Expedited pathway reduces review by ~40% in most LATAM countries
                min_days = int(min_days * 0.6)
                max_days = int(max_days * 0.6)

            roadmap["countries"][country] = {
                "agency": profile["agency"],
                "language": profile["language"],
                "submission_system": profile["system"],
                "key_regulations": profile["key_regulations"],
                "estimated_review_days": f"{min_days}–{max_days}",
                "expedited_pathway_available": expedited,
                "priority_pathway_note": profile["priority_pathway"] if expedited else None,
                "dossier_language": profile["language"],
            }

            priority_order.append((country, min_days))

        # Recommend sequence: fastest approval first to build market momentum
        priority_order.sort(key=lambda x: x[1])
        roadmap["recommended_sequence"] = [c for c, _ in priority_order]

        if priority_order:
            roadmap["total_estimated_timeline_months"] = round(
                max(d for _, d in priority_order) / 30, 1
            )

        return roadmap

    def build_submission_prompt(
        self,
        country: str,
        product_description: str,
        existing_approvals: Optional[list[str]] = None,
    ) -> str:
        """
        Build a regulatory prompt for the LATAM layer engine.
        """
        profile = self.get_agency_profile(country)
        if not profile:
            return (
                f"Provide regulatory guidance for submitting '{product_description}' "
                f"to Latin American authorities. List countries and their key requirements."
            )

        approvals_str = ", ".join(existing_approvals) if existing_approvals else "none"

        return (
            f"LATAM REGULATORY ANALYSIS REQUEST\n"
            f"Country: {country.upper()} | Agency: {profile['agency']}\n"
            f"Product: {product_description}\n"
            f"Existing Reference Approvals: {approvals_str}\n"
            f"Applicable Regulations: {', '.join(profile['key_regulations'])}\n"
            f"Submission System: {profile['system']}\n"
            f"Dossier Language: {profile['language']}\n\n"
            f"Please provide:\n"
            f"1. Recommended submission pathway (standard vs expedited)\n"
            f"2. Key dossier modules required per {profile['agency']} guidelines\n"
            f"3. Bioequivalence data requirements\n"
            f"4. Estimated timeline and review stages\n"
            f"5. Country-specific risks and mitigation strategies\n"
            f"6. Post-marketing pharmacovigilance obligations\n"
        )

    def format_harmonization_context(self) -> str:
        """Return PANDRH/MERCOSUR harmonization context for RAG enrichment."""
        return (
            "LATAM REGULATORY HARMONIZATION CONTEXT:\n"
            + "\n".join(f"- {f}" for f in HARMONIZATION_FRAMEWORKS)
            + "\n\nKey principle: Where a product holds FDA, EMA, or WHO Prequalification, "
            "most LATAM agencies offer abbreviated or recognition-based pathways. "
            "Brazil (ANVISA) and Colombia (INVIMA) are the most ICH-aligned. "
            "Argentina (ANMAT) and Mexico (COFEPRIS) have unique national requirements "
            "that may require additional local studies."
        )


latam_agent = LatamRegulatoryAgent()
