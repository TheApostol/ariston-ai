"""
PubMed E-utilities source.
Fetches abstracts from PubMed (40M+ citations) — free, no API key required.
"""

import httpx
from typing import List

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


async def search_pubmed(query: str, max_results: int = 5) -> List[dict]:
    """Return list of {title, abstract, pmid, source} from PubMed."""
    async with httpx.AsyncClient(timeout=15) as client:
        # Step 1: get PMIDs
        search_resp = await client.get(ESEARCH_URL, params={
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "sort": "relevance",
        })
        search_resp.raise_for_status()
        ids = search_resp.json().get("esearchresult", {}).get("idlist", [])

        if not ids:
            return []

        # Step 2: fetch abstracts
        fetch_resp = await client.get(EFETCH_URL, params={
            "db": "pubmed",
            "id": ",".join(ids),
            "rettype": "abstract",
            "retmode": "text",
        })
        fetch_resp.raise_for_status()
        raw = fetch_resp.text

    # Split into per-abstract chunks
    chunks = [c.strip() for c in raw.split("\n\n\n") if c.strip()]
    results = []
    for i, chunk in enumerate(chunks[:max_results]):
        results.append({
            "source": "pubmed",
            "pmid": ids[i] if i < len(ids) else None,
            "content": chunk[:1500],  # cap per chunk
        })
    return results
