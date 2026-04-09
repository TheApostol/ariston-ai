"""
OMOP CDM / OHDSI vocabulary mapping.
Industry standard for observational health data — required for any real pharma pilot.
Uses OHDSI public Athena vocabulary API (free, no key).

Maps: drug names → RxNorm concept IDs
      conditions → SNOMED-CT concept IDs
      procedures → CPT/SNOMED IDs
      labs → LOINC concept IDs
"""

import httpx
from typing import Dict, List, Optional

ATHENA_URL = "https://athena.ohdsi.org/api/v1/concepts"
RXNORM_URL = "https://rxnav.nlm.nih.gov/REST"


# Standard OMOP vocabulary IDs
VOCAB = {
    "drug":      "RxNorm",
    "condition": "SNOMED",
    "procedure": "SNOMED",
    "lab":       "LOINC",
    "device":    "SNOMED",
}


async def map_to_omop(term: str, domain: str = "drug") -> Optional[Dict]:
    """
    Map a clinical term to its OMOP standard concept.
    Returns concept_id, concept_name, vocabulary_id, domain.
    """
    vocab = VOCAB.get(domain, "RxNorm")
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            r = await client.get(ATHENA_URL, params={
                "query": term,
                "vocabularyId": vocab,
                "pageSize": 1,
                "page": 1,
            })
            if r.status_code != 200:
                return await _fallback_rxnorm(term, client)

            concepts = r.json().get("content", [])
            if not concepts:
                return await _fallback_rxnorm(term, client)

            c = concepts[0]
            return {
                "concept_id": c.get("id"),
                "concept_name": c.get("name"),
                "vocabulary": c.get("vocabularyId"),
                "domain": c.get("domainId"),
                "standard": c.get("standardConcept") == "Standard",
                "source": "omop_athena",
            }
        except Exception as e:
            print(f"[OMOP] {e}")
            return None


async def _fallback_rxnorm(term: str, client: httpx.AsyncClient) -> Optional[Dict]:
    """Fallback: resolve via RxNorm when Athena is unavailable."""
    try:
        r = await client.get(f"{RXNORM_URL}/rxcui.json", params={"name": term})
        ids = r.json().get("idGroup", {}).get("rxnormId", [])
        if not ids:
            return None
        return {
            "concept_id": ids[0],
            "concept_name": term,
            "vocabulary": "RxNorm",
            "domain": "Drug",
            "standard": True,
            "source": "rxnorm_fallback",
        }
    except Exception:
        return None


async def map_batch(terms: List[str], domain: str = "drug") -> List[Dict]:
    """Map multiple terms to OMOP concepts."""
    import asyncio
    results = await asyncio.gather(*[map_to_omop(t, domain) for t in terms], return_exceptions=True)
    return [r for r in results if isinstance(r, dict) and r is not None]


def format_omop_context(concepts: List[Dict]) -> str:
    """Format OMOP concepts as context string for LLM injection."""
    if not concepts:
        return ""
    lines = ["OMOP Standard Concepts:"]
    for c in concepts:
        lines.append(f"  [{c.get('vocabulary')}:{c.get('concept_id')}] {c.get('concept_name')} ({c.get('domain')})")
    return "\n".join(lines)
