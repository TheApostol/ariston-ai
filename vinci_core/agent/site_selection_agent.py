"""
Clinical Trial Site Selection Agent for LatAm.

Ranks and recommends clinical trial sites across Latin America based on:
  - Regulatory agency alignment (ANVISA, COFEPRIS, INVIMA, ANMAT)
  - Therapeutic area expertise
  - Patient pool size (therapeutic area prevalence)
  - Investigator experience tier
  - Regulatory track record (approval speed)
  - Site capacity and infrastructure score

Outputs scored site recommendations compatible with the ICH E6(R3) GCP
site selection guidance and LatAm-specific regulatory requirements.
"""

from typing import Any, Dict, List, Optional


# ── Static site knowledge base ────────────────────────────────────────────────
# In production this would be backed by a live database / CRO partner API.

LATAM_SITES: List[Dict[str, Any]] = [
    {
        "site_id": "BR-SP-001",
        "name": "Hospital das Clínicas da FMUSP",
        "country": "Brazil",
        "city": "São Paulo",
        "locale": "pt-BR",
        "agency": "ANVISA",
        "therapeutic_areas": ["oncology", "cardiology", "neurology", "rare_disease"],
        "patient_pool_estimate": 50000,
        "investigator_tier": "tier1",
        "avg_approval_days": 90,
        "capacity_score": 0.95,
        "gcp_certification": True,
        "contact": "pesquisa@hc.fm.usp.br",
    },
    {
        "site_id": "BR-RJ-001",
        "name": "INCA — Instituto Nacional do Câncer",
        "country": "Brazil",
        "city": "Rio de Janeiro",
        "locale": "pt-BR",
        "agency": "ANVISA",
        "therapeutic_areas": ["oncology", "hematology"],
        "patient_pool_estimate": 30000,
        "investigator_tier": "tier1",
        "avg_approval_days": 95,
        "capacity_score": 0.88,
        "gcp_certification": True,
        "contact": "pesquisa@inca.gov.br",
    },
    {
        "site_id": "MX-CDMX-001",
        "name": "Instituto Nacional de Cancerología (INCan)",
        "country": "Mexico",
        "city": "Mexico City",
        "locale": "es-MX",
        "agency": "COFEPRIS",
        "therapeutic_areas": ["oncology", "hematology", "immunology"],
        "patient_pool_estimate": 20000,
        "investigator_tier": "tier1",
        "avg_approval_days": 120,
        "capacity_score": 0.85,
        "gcp_certification": True,
        "contact": "investigacion@incan.gob.mx",
    },
    {
        "site_id": "MX-CDMX-002",
        "name": "Instituto Nacional de Cardiología Ignacio Chávez",
        "country": "Mexico",
        "city": "Mexico City",
        "locale": "es-MX",
        "agency": "COFEPRIS",
        "therapeutic_areas": ["cardiology", "metabolic", "hypertension"],
        "patient_pool_estimate": 15000,
        "investigator_tier": "tier1",
        "avg_approval_days": 115,
        "capacity_score": 0.82,
        "gcp_certification": True,
        "contact": "investigacion@cardiologia.org.mx",
    },
    {
        "site_id": "CO-BOG-001",
        "name": "Clínica del Country — Bogotá Research Unit",
        "country": "Colombia",
        "city": "Bogotá",
        "locale": "es-CO",
        "agency": "INVIMA",
        "therapeutic_areas": ["oncology", "infectious_disease", "rheumatology"],
        "patient_pool_estimate": 10000,
        "investigator_tier": "tier2",
        "avg_approval_days": 150,
        "capacity_score": 0.75,
        "gcp_certification": True,
        "contact": "investigacion@clinicadelcountry.com",
    },
    {
        "site_id": "CO-MED-001",
        "name": "Hospital Pablo Tobón Uribe — Medellín",
        "country": "Colombia",
        "city": "Medellín",
        "locale": "es-CO",
        "agency": "INVIMA",
        "therapeutic_areas": ["gastroenterology", "hepatology", "transplant"],
        "patient_pool_estimate": 8000,
        "investigator_tier": "tier2",
        "avg_approval_days": 155,
        "capacity_score": 0.72,
        "gcp_certification": True,
        "contact": "investigacion@hptu.org.co",
    },
    {
        "site_id": "AR-BA-001",
        "name": "Hospital Italiano de Buenos Aires — Research Division",
        "country": "Argentina",
        "city": "Buenos Aires",
        "locale": "es-AR",
        "agency": "ANMAT",
        "therapeutic_areas": ["oncology", "cardiology", "endocrinology", "neurology"],
        "patient_pool_estimate": 18000,
        "investigator_tier": "tier1",
        "avg_approval_days": 100,
        "capacity_score": 0.90,
        "gcp_certification": True,
        "contact": "investigacion@hospitalitaliano.org.ar",
    },
    {
        "site_id": "AR-BA-002",
        "name": "Instituto Ángel H. Roffo — Oncología",
        "country": "Argentina",
        "city": "Buenos Aires",
        "locale": "es-AR",
        "agency": "ANMAT",
        "therapeutic_areas": ["oncology", "hematology", "radiation_oncology"],
        "patient_pool_estimate": 12000,
        "investigator_tier": "tier1",
        "avg_approval_days": 105,
        "capacity_score": 0.83,
        "gcp_certification": True,
        "contact": "investigacion@roffo.gov.ar",
    },
]

# Tier weights for scoring
_TIER_SCORE = {"tier1": 1.0, "tier2": 0.75, "tier3": 0.50}


class SiteSelectionAgent:
    """
    Scores and ranks LatAm clinical trial sites for a given trial profile.

    Scoring formula (100-point scale):
      - Therapeutic area match          : 30 pts
      - Capacity score                  : 25 pts
      - Investigator tier               : 20 pts
      - Approval speed (inverse)        : 15 pts  (faster = higher)
      - Patient pool (log-normalized)   : 10 pts
    """

    MAX_APPROVAL_DAYS = 200   # normalisation ceiling

    def score_site(
        self, site: Dict[str, Any], therapeutic_area: str, agency: Optional[str] = None
    ) -> float:
        """Return a 0-100 composite score for a site."""
        ta_match = 30.0 if therapeutic_area.lower() in [
            t.lower() for t in site.get("therapeutic_areas", [])
        ] else 0.0

        capacity = site.get("capacity_score", 0.5) * 25.0
        tier = _TIER_SCORE.get(site.get("investigator_tier", "tier3"), 0.5) * 20.0

        raw_days = site.get("avg_approval_days", self.MAX_APPROVAL_DAYS)
        speed = max(0.0, (1.0 - raw_days / self.MAX_APPROVAL_DAYS)) * 15.0

        import math
        pool = site.get("patient_pool_estimate", 1)
        pool_score = min(10.0, math.log10(max(1, pool)) * 2.5)

        return round(ta_match + capacity + tier + speed + pool_score, 2)

    def recommend_sites(
        self,
        therapeutic_area: str,
        agency: Optional[str] = None,
        country: Optional[str] = None,
        top_n: int = 5,
        min_score: float = 40.0,
    ) -> Dict[str, Any]:
        """
        Recommend and rank the best LatAm sites for a trial.

        Args:
            therapeutic_area: e.g. "oncology", "cardiology"
            agency:           optional filter by regulatory agency (ANVISA/COFEPRIS/INVIMA/ANMAT)
            country:          optional filter by country
            top_n:            maximum number of sites to return
            min_score:        minimum composite score threshold (0-100)

        Returns:
            Dict with ranked site recommendations and composite scores.
        """
        candidates = LATAM_SITES

        if agency:
            candidates = [s for s in candidates if s["agency"].upper() == agency.upper()]
        if country:
            candidates = [s for s in candidates if s["country"].lower() == country.lower()]

        scored = []
        for site in candidates:
            score = self.score_site(site, therapeutic_area, agency)
            if score >= min_score:
                scored.append({**site, "composite_score": score})

        scored.sort(key=lambda x: x["composite_score"], reverse=True)
        ranked = scored[:top_n]

        # Tag rank
        for i, s in enumerate(ranked, 1):
            s["rank"] = i

        return {
            "therapeutic_area": therapeutic_area,
            "agency_filter": agency,
            "country_filter": country,
            "sites_evaluated": len(candidates),
            "sites_recommended": len(ranked),
            "recommendations": ranked,
        }

    def feasibility_summary(
        self, therapeutic_area: str, agencies: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate a multi-country feasibility overview.

        Returns per-agency top site + overall readiness tier.
        """
        target_agencies = agencies or ["ANVISA", "COFEPRIS", "INVIMA", "ANMAT"]
        per_agency = {}
        for ag in target_agencies:
            result = self.recommend_sites(
                therapeutic_area=therapeutic_area,
                agency=ag,
                top_n=1,
                min_score=0.0,
            )
            top = result["recommendations"][0] if result["recommendations"] else None
            per_agency[ag] = {
                "top_site": top.get("name") if top else None,
                "top_score": top.get("composite_score") if top else None,
                "country": top.get("country") if top else None,
                "avg_approval_days": top.get("avg_approval_days") if top else None,
            }

        scores = [v["top_score"] for v in per_agency.values() if v["top_score"] is not None]
        avg_score = round(sum(scores) / len(scores), 1) if scores else 0.0
        readiness = "HIGH" if avg_score >= 70 else "MODERATE" if avg_score >= 50 else "LOW"

        return {
            "therapeutic_area": therapeutic_area,
            "overall_readiness": readiness,
            "avg_composite_score": avg_score,
            "per_agency": per_agency,
        }


site_selection_agent = SiteSelectionAgent()
