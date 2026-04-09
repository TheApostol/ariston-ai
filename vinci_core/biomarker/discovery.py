"""
Biomarker Discovery Pipeline — Phase 3 / Ariston AI.

AI-assisted biomarker hypothesis generation from:
  - RWE cohort data (accumulated in Phase 2)
  - PubMed literature (RAG-enriched)
  - Genomic variant databases (future: OpenTargets, GWAS Catalog)
  - LATAM-specific disease burden data (PAHO/WHO)

Phase 3 goal: "Add drug discovery AI tools leveraging proprietary data,
clinical decision support for providers, and international expansion."

Biomarker types supported:
  - Prognostic: predict disease course
  - Predictive: predict treatment response
  - Pharmacodynamic: measure drug effect
  - Safety/Toxicity: predict adverse reactions (PGx integration)
  - Companion diagnostic: eligibility for targeted therapy

The data flywheel: RWE data (Phase 2) → Biomarker discovery → Better clinical
trial design → More data → Better biomarkers → Platform lock-in.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("ariston.biomarker")

BIOMARKER_TYPES = [
    "prognostic",
    "predictive",
    "pharmacodynamic",
    "safety_toxicity",
    "companion_diagnostic",
    "surrogate_endpoint",
]

LATAM_DISEASE_PRIORITIES = {
    "brazil": [
        "dengue", "chagas_disease", "leishmaniasis", "zika",
        "type2_diabetes", "cardiovascular", "oncology_gastric",
    ],
    "mexico": [
        "type2_diabetes", "obesity", "cardiovascular", "cervical_cancer",
        "dengue", "hepatitis_c",
    ],
    "colombia": [
        "malaria", "dengue", "chagas_disease", "tuberculosis",
        "leishmania", "oncology_gastric", "cardiovascular",
    ],
    "argentina": [
        "cardiovascular", "oncology_breast", "oncology_colorectal",
        "type2_diabetes", "chagas_disease", "hepatitis_b",
    ],
    "chile": [
        "oncology_gastric", "cardiovascular", "type2_diabetes",
        "alzheimer", "copd", "oncology_breast",
    ],
}


@dataclass
class BiomarkerHypothesis:
    """A candidate biomarker with supporting evidence and confidence score."""
    hypothesis_id: str
    biomarker_name: str
    biomarker_type: str
    disease_area: str
    mechanism: str
    evidence_level: str  # preclinical | early_clinical | validated
    confidence_score: float  # 0.0–1.0
    supporting_literature: list[str] = field(default_factory=list)
    latam_relevance: dict = field(default_factory=dict)
    regulatory_path: str = ""
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: dict = field(default_factory=dict)


class BiomarkerDiscoveryEngine:
    """
    Phase 3 biomarker hypothesis generator.

    Combines literature-based evidence (PubMed RAG) with RWE cohort signals
    and LATAM disease burden data to prioritize biomarker candidates.
    """

    async def generate_hypotheses(
        self,
        disease_area: str,
        biomarker_type: str = "predictive",
        countries: Optional[list[str]] = None,
        use_rwe: bool = False,
    ) -> list[BiomarkerHypothesis]:
        """
        Generate biomarker hypotheses for a disease area.

        Args:
            disease_area: therapeutic area (e.g. "type2_diabetes")
            biomarker_type: type from BIOMARKER_TYPES
            countries: LATAM markets to consider for disease burden context
            use_rwe: whether to incorporate RWE data signals

        Returns:
            List of BiomarkerHypothesis candidates ranked by confidence
        """
        from vinci_core.engine import engine

        countries = countries or []
        latam_context = self._build_latam_context(disease_area, countries)

        prompt = (
            f"BIOMARKER DISCOVERY ANALYSIS\n"
            f"Disease Area: {disease_area}\n"
            f"Biomarker Type: {biomarker_type}\n"
            f"LATAM Markets: {', '.join(c.upper() for c in countries) or 'Global'}\n"
            f"RWE Data Available: {'Yes' if use_rwe else 'Literature only'}\n"
            f"\n{latam_context}\n\n"
            f"Generate 3 candidate {biomarker_type} biomarker hypotheses for {disease_area}.\n"
            f"For each biomarker provide:\n"
            f"1. Biomarker name and molecular target\n"
            f"2. Proposed mechanism of action\n"
            f"3. Evidence level (preclinical/early_clinical/validated)\n"
            f"4. Confidence assessment (0-100%)\n"
            f"5. Recommended assay or measurement method\n"
            f"6. Regulatory path to companion diagnostic (if applicable)\n"
            f"7. LATAM-specific considerations (disease burden, population genetics)\n"
            f"\nInclude uncertainty language. These are hypotheses requiring validation."
        )

        response = await engine.run(
            prompt=prompt,
            layer="pharma",
            use_rag=True,
        )

        # Parse AI response into structured hypothesis (simplified)
        hypothesis = BiomarkerHypothesis(
            hypothesis_id=f"BM-{disease_area[:4].upper()}-001",
            biomarker_name=f"{disease_area} candidate biomarker",
            biomarker_type=biomarker_type,
            disease_area=disease_area,
            mechanism="See AI analysis below",
            evidence_level="preclinical",
            confidence_score=0.60,
            latam_relevance={c: LATAM_DISEASE_PRIORITIES.get(c, []) for c in countries},
            metadata={"ai_analysis": response.content},
        )

        logger.info(
            "[Biomarker] generated hypothesis_id=%s disease=%s type=%s",
            hypothesis.hypothesis_id, disease_area, biomarker_type,
        )
        return [hypothesis]

    def _build_latam_context(self, disease_area: str, countries: list[str]) -> str:
        if not countries:
            return ""
        context = "LATAM DISEASE BURDEN CONTEXT:\n"
        for country in countries:
            priorities = LATAM_DISEASE_PRIORITIES.get(country.lower(), [])
            if disease_area.lower() in [p.lower() for p in priorities]:
                context += f"- {country.upper()}: {disease_area} is a priority disease area\n"
            else:
                context += f"- {country.upper()}: {disease_area} context — verify local prevalence\n"
        return context

    def get_latam_disease_priorities(self, country: str = None) -> dict:
        if country:
            return {country.lower(): LATAM_DISEASE_PRIORITIES.get(country.lower(), [])}
        return LATAM_DISEASE_PRIORITIES


biomarker_engine = BiomarkerDiscoveryEngine()
