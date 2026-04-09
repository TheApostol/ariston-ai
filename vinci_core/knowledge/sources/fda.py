"""
FDA OpenFDA API source.
Covers drug labels, adverse events, and enforcement data — free, no key required.
"""

import httpx
from typing import List

DRUG_LABEL_URL = "https://api.fda.gov/drug/label.json"
ADVERSE_EVENT_URL = "https://api.fda.gov/drug/event.json"


async def search_drug_labels(query: str, max_results: int = 3) -> List[dict]:
    """Search FDA drug labels for a compound or condition."""
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(DRUG_LABEL_URL, params={
            "search": query,
            "limit": max_results,
        })
        if resp.status_code != 200:
            return []
        results_raw = resp.json().get("results", [])

    results = []
    for r in results_raw:
        content_parts = []
        for field in ("indications_and_usage", "warnings", "dosage_and_administration", "description"):
            val = r.get(field)
            if val:
                text = val[0] if isinstance(val, list) else val
                content_parts.append(f"{field.upper()}:\n{text[:400]}")

        results.append({
            "source": "fda_label",
            "brand_name": (r.get("openfda", {}).get("brand_name") or ["unknown"])[0],
            "content": "\n\n".join(content_parts)[:1500],
        })
    return results


async def search_adverse_events(drug_name: str, max_results: int = 3) -> List[dict]:
    """Retrieve adverse event summaries for a drug from FAERS."""
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(ADVERSE_EVENT_URL, params={
            "search": f"patient.drug.medicinalproduct:{drug_name}",
            "limit": max_results,
        })
        if resp.status_code != 200:
            return []
        results_raw = resp.json().get("results", [])

    results = []
    for r in results_raw:
        reactions = [
            rx.get("reactionmeddrapt", "")
            for rx in r.get("patient", {}).get("reaction", [])
        ]
        results.append({
            "source": "fda_faers",
            "serious": r.get("serious"),
            "reactions": reactions[:10],
            "content": f"Adverse reactions reported: {', '.join(reactions[:10])}",
        })
    return results
