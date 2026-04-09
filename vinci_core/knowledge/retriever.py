"""
RAG retriever — layer-routed fetching across all knowledge sources + ChromaDB cache.

pharma   → FDA labels + PubMed + DrugBank interactions + OpenTargets
clinical → PubMed + ClinicalTrials.gov
data     → PubMed + ClinicalTrials.gov + OpenTargets
base     → PubMed only
"""

import asyncio
from typing import List
from vinci_core.knowledge.store import upsert_documents, query_store
from vinci_core.knowledge.sources.pubmed import search_pubmed
from vinci_core.knowledge.sources.clinicaltrials import search_trials
from vinci_core.knowledge.sources.fda import search_drug_labels, search_adverse_events
from vinci_core.knowledge.sources.drugbank import get_drug_interactions, get_drug_targets
from vinci_core.knowledge.sources.opentargets import get_drug_disease_associations


async def retrieve(query: str, layer: str = "base", max_results: int = 5) -> List[dict]:
    # Fast path: vector store cache
    cached = query_store(query, n_results=max_results)
    if len(cached) >= max_results:
        return cached

    # Layer-routed live fetches
    fetch_tasks = []

    if layer == "pharma":
        fetch_tasks += [
            search_drug_labels(query, max_results=3),
            search_adverse_events(query, max_results=2),
            search_pubmed(query + " regulatory pharma", max_results=3),
            get_drug_interactions(query),
            get_drug_targets(query),
            get_drug_disease_associations(query),
        ]
    elif layer == "clinical":
        fetch_tasks += [
            search_pubmed(query + " clinical", max_results=4),
            search_trials(query, max_results=3),
        ]
    elif layer == "data":
        fetch_tasks += [
            search_pubmed(query + " biomarker real-world evidence", max_results=4),
            search_trials(query, max_results=3),
            get_drug_disease_associations(query),
        ]
    else:
        fetch_tasks += [search_pubmed(query, max_results=5)]

    fetched_groups = await asyncio.gather(*fetch_tasks, return_exceptions=True)

    all_docs = []
    for group in fetched_groups:
        if isinstance(group, (Exception, BaseException)):
            continue
        if isinstance(group, list):
            all_docs.extend(group)
        elif isinstance(group, dict):
            all_docs.append(group)

    if all_docs:
        upsert_documents(all_docs)

    return query_store(query, n_results=max_results) or all_docs[:max_results]


def format_context(chunks: List[dict]) -> str:
    if not chunks:
        return ""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("source", "unknown")
        content = chunk.get("content", "")
        if content:
            parts.append(f"[Source {i} — {source}]\n{content}")
    return "\n\n---\n\n".join(parts)
