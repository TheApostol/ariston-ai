"""
ClinicalTrials.gov API v2 source.
Free, no API key required.
"""

import httpx
from typing import List

BASE_URL = "https://clinicaltrials.gov/api/v2/studies"


async def search_trials(query: str, max_results: int = 5) -> List[dict]:
    """Return list of {nctId, title, status, phase, brief_summary, source}."""
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(BASE_URL, params={
            "query.term": query,
            "pageSize": max_results,
            "format": "json",
            "fields": "NCTId,BriefTitle,OverallStatus,Phase,BriefSummary",
        })
        resp.raise_for_status()
        studies = resp.json().get("studies", [])

    results = []
    for s in studies:
        proto = s.get("protocolSection", {})
        id_mod = proto.get("identificationModule", {})
        status_mod = proto.get("statusModule", {})
        desc_mod = proto.get("descriptionModule", {})
        design_mod = proto.get("designModule", {})

        results.append({
            "source": "clinicaltrials",
            "nct_id": id_mod.get("nctId"),
            "title": id_mod.get("briefTitle"),
            "status": status_mod.get("overallStatus"),
            "phase": design_mod.get("phases", []),
            "content": desc_mod.get("briefSummary", "")[:1500],
        })
    return results
