"""
PubMed E-utilities source.
Fetches abstracts from PubMed (40M+ citations) — free, no API key required.

Improvements:
  - 5 s timeout with graceful fallback to simulated context on timeout
  - SQLite 24-hour cache to avoid repeat identical queries
  - Claude Haiku post-processing: raw abstracts are summarised into 2-3 grounded sentences
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import time
from typing import List

import httpx

logger = logging.getLogger("ariston.pubmed")

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

_PUBMED_TIMEOUT_S = 5          # hard timeout for each PubMed call
_CACHE_TTL_S      = 86_400     # 24 h
_CACHE_DB         = os.path.join("data", "ariston.db")

# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_key(query: str, max_results: int) -> str:
    return hashlib.sha256(f"pubmed:{query}:{max_results}".encode()).hexdigest()


def _cache_init() -> None:
    try:
        with sqlite3.connect(_CACHE_DB) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS pubmed_cache (
                    cache_key   TEXT PRIMARY KEY,
                    result_json TEXT NOT NULL,
                    created_at  INTEGER NOT NULL
                )
            """)
            conn.commit()
    except Exception as e:
        logger.debug("[PubMed] cache init error: %s", e)


def _cache_get(key: str) -> List[dict] | None:
    try:
        with sqlite3.connect(_CACHE_DB) as conn:
            row = conn.execute(
                "SELECT result_json, created_at FROM pubmed_cache WHERE cache_key=?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        created_at = row[1]
        if time.time() - created_at > _CACHE_TTL_S:
            return None   # stale
        return json.loads(row[0])
    except Exception as e:
        logger.debug("[PubMed] cache read error: %s", e)
        return None


def _cache_set(key: str, results: List[dict]) -> None:
    try:
        with sqlite3.connect(_CACHE_DB) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO pubmed_cache (cache_key, result_json, created_at) VALUES (?,?,?)",
                (key, json.dumps(results), int(time.time())),
            )
            conn.commit()
    except Exception as e:
        logger.debug("[PubMed] cache write error: %s", e)


# ---------------------------------------------------------------------------
# Haiku summarization
# ---------------------------------------------------------------------------

async def _summarise_with_haiku(abstracts_text: str, query: str) -> str:
    """Summarise retrieved PubMed abstracts into 2-3 grounded sentences using Claude Haiku."""
    try:
        import anthropic as _anthropic
        client = _anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        prompt = (
            f"You are a medical literature analyst. Summarise the following PubMed abstracts "
            f"into 2-3 concise, evidence-grounded sentences relevant to the query: '{query}'.\n\n"
            f"Abstracts:\n{abstracts_text[:3000]}\n\n"
            f"Provide only the summary, no preamble."
        )
        message = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text.strip()
    except Exception as e:
        logger.debug("[PubMed] Haiku summarisation failed: %s", e)
        return ""


# ---------------------------------------------------------------------------
# Simulated fallback context
# ---------------------------------------------------------------------------

_SIMULATED_CONTEXT = [
    {
        "source": "pubmed_simulated",
        "pmid": None,
        "content": (
            "[Simulated context — PubMed timed out] Current evidence suggests that clinical "
            "decision support should be grounded in peer-reviewed literature. Consult primary "
            "sources for definitive guidance."
        ),
    }
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_cache_init()


async def search_pubmed(
    query: str,
    max_results: int = 5,
    summarise: bool = True,
) -> List[dict]:
    """
    Return list of {source, pmid, content} from PubMed.

    Behaviour:
      - Checks 24-hour SQLite cache first.
      - Calls PubMed E-utilities with a 5 s hard timeout.
      - Falls back to _SIMULATED_CONTEXT on timeout or network error.
      - Optionally calls Claude Haiku to produce a 2-3 sentence summary
        and prepends it as an extra chunk.
    """
    key = _cache_key(query, max_results)

    # 1. Cache hit
    cached = _cache_get(key)
    if cached is not None:
        logger.debug("[PubMed] cache hit for query='%.60s'", query)
        return cached

    # 2. Live fetch
    try:
        async with httpx.AsyncClient(timeout=_PUBMED_TIMEOUT_S) as client:
            search_resp = await client.get(ESEARCH_URL, params={
                "db":      "pubmed",
                "term":    query,
                "retmax":  max_results,
                "retmode": "json",
                "sort":    "relevance",
            })
            search_resp.raise_for_status()
            ids = search_resp.json().get("esearchresult", {}).get("idlist", [])

            if not ids:
                _cache_set(key, [])
                return []

            fetch_resp = await client.get(EFETCH_URL, params={
                "db":      "pubmed",
                "id":      ",".join(ids),
                "rettype": "abstract",
                "retmode": "text",
            })
            fetch_resp.raise_for_status()
            raw = fetch_resp.text

    except (httpx.TimeoutException, httpx.ConnectError, httpx.ReadTimeout) as e:
        logger.warning("[PubMed] timeout/connect error — using simulated context: %s", e)
        return _SIMULATED_CONTEXT
    except Exception as e:
        logger.warning("[PubMed] fetch error — using simulated context: %s", e)
        return _SIMULATED_CONTEXT

    # 3. Parse
    chunks   = [c.strip() for c in raw.split("\n\n\n") if c.strip()]
    results  = []
    for i, chunk in enumerate(chunks[:max_results]):
        results.append({
            "source":  "pubmed",
            "pmid":    ids[i] if i < len(ids) else None,
            "content": chunk[:1500],
        })

    # 4. Claude Haiku summarisation
    if summarise and results:
        combined = "\n\n".join(r["content"] for r in results)
        summary  = await _summarise_with_haiku(combined, query)
        if summary:
            results.insert(0, {
                "source":  "pubmed_summary",
                "pmid":    None,
                "content": summary,
            })

    # 5. Cache and return
    _cache_set(key, results)
    return results
