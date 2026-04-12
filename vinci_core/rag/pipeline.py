"""
RAG Pipeline — Ariston AI.

Production-grade Retrieval-Augmented Generation for life sciences:
  1. Query expansion     — synonym + MeSH term expansion for biomedical queries
  2. Multi-source fetch  — PubMed + OpenTargets + ClinicalTrials.gov in parallel
  3. Chunk scoring       — TF-IDF-style relevance scoring (no GPU required)
  4. Context assembly    — ranked chunks → LLM context window
  5. Source citation     — every answer traceable to source + PMID

This replaces the stub `use_rag=True` calls throughout the codebase.
The engine's `run()` method routes rag=True calls through here.

Production upgrade path:
  - Swap chunk scorer → sentence-transformers (all-MiniLM-L6-v2)
  - Add ChromaDB / pgvector for persistent chunk storage
  - Add FAISS for approximate nearest-neighbour retrieval at scale
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("ariston.rag")

# MeSH-style synonym expansion for common biomedical terms
_MESH_SYNONYMS: dict[str, list[str]] = {
    "type2 diabetes": ["T2DM", "type 2 diabetes mellitus", "NIDDM", "insulin resistance"],
    "chagas":         ["Trypanosoma cruzi", "American trypanosomiasis", "chagasic cardiomyopathy"],
    "dengue":         ["DENV", "dengue fever", "dengue hemorrhagic fever", "dengue virus"],
    "cardiovascular": ["CVD", "coronary artery disease", "heart failure", "atherosclerosis"],
    "oncology":       ["cancer", "tumor", "malignancy", "neoplasm", "carcinoma"],
    "biomarker":      ["biological marker", "molecular marker", "prognostic marker"],
    "pharmacovigilance": ["adverse event", "ADR", "drug safety", "post-marketing surveillance"],
    "latam":          ["Latin America", "South America", "Brazil", "Mexico", "Colombia"],
}


@dataclass
class RAGChunk:
    """A retrieved text chunk with metadata."""
    content: str
    source: str          # pubmed | opentargets | clinicaltrials | internal
    reference_id: str    # PMID, K-number, NCT ID, etc.
    relevance_score: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class RAGResult:
    """Assembled RAG context ready for LLM injection."""
    context_text: str
    chunks: list[RAGChunk]
    sources_cited: list[str]
    query_expanded: str
    total_chunks_retrieved: int
    total_chunks_used: int


class RAGPipeline:
    """
    Multi-source RAG pipeline for life sciences queries.

    Retrieves from PubMed, OpenTargets, and ClinicalTrials.gov in parallel,
    scores chunks by relevance, and assembles a context window for the LLM.
    """

    def __init__(
        self,
        max_chunks: int = 6,
        max_chars_per_chunk: int = 800,
        context_budget: int = 4000,
    ):
        self.max_chunks = max_chunks
        self.max_chars_per_chunk = max_chars_per_chunk
        self.context_budget = context_budget

    async def retrieve(
        self,
        query: str,
        layer: str = "pharma",
        use_pubmed: bool = True,
        use_opentargets: bool = True,
        use_clinicaltrials: bool = False,
        country_filter: Optional[str] = None,
    ) -> RAGResult:
        """
        Retrieve and rank relevant chunks for a query.

        Args:
            query: natural language or structured biomedical query
            layer: clinical | pharma | latam — affects source weighting
            use_pubmed: include PubMed abstracts
            use_opentargets: include Open Targets drug/target data
            use_clinicaltrials: include ClinicalTrials.gov data
            country_filter: LATAM country filter for trial/RWE sources

        Returns:
            RAGResult with ranked context ready for LLM injection
        """
        expanded_query = self._expand_query(query)

        # Parallel fetch from all sources
        fetch_tasks = []
        if use_pubmed:
            fetch_tasks.append(self._fetch_pubmed(expanded_query))
        if use_opentargets:
            fetch_tasks.append(self._fetch_opentargets(query))
        if use_clinicaltrials:
            fetch_tasks.append(self._fetch_clinicaltrials(query, country_filter))

        if not fetch_tasks:
            return RAGResult("", [], [], expanded_query, 0, 0)

        results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

        all_chunks: list[RAGChunk] = []
        for r in results:
            if isinstance(r, list):
                all_chunks.extend(r)
            elif isinstance(r, Exception):
                logger.debug("[RAG] Source fetch error: %s", r)

        # Score and rank
        scored = self._score_chunks(all_chunks, query)
        top_chunks = scored[:self.max_chunks]

        # Assemble context
        context_text = self._assemble_context(top_chunks)
        sources = list({c.source for c in top_chunks})

        logger.info(
            "[RAG] query_len=%d expanded=%d retrieved=%d used=%d sources=%s",
            len(query), len(expanded_query), len(all_chunks), len(top_chunks), sources,
        )

        return RAGResult(
            context_text=context_text,
            chunks=top_chunks,
            sources_cited=sources,
            query_expanded=expanded_query,
            total_chunks_retrieved=len(all_chunks),
            total_chunks_used=len(top_chunks),
        )

    def build_prompt_with_context(self, original_prompt: str, rag_result: RAGResult) -> str:
        """Inject RAG context into the LLM prompt."""
        if not rag_result.context_text:
            return original_prompt

        return (
            f"RETRIEVED EVIDENCE (from {', '.join(rag_result.sources_cited)}):\n"
            f"{'─'*60}\n"
            f"{rag_result.context_text}\n"
            f"{'─'*60}\n\n"
            f"USER QUERY:\n{original_prompt}\n\n"
            f"Instructions: Base your answer on the retrieved evidence above. "
            f"Cite sources by reference ID where possible. "
            f"Acknowledge if evidence is insufficient or contradictory."
        )

    # ---------------------------------------------------------------------------
    # Source fetchers
    # ---------------------------------------------------------------------------

    async def _fetch_pubmed(self, query: str) -> list[RAGChunk]:
        from vinci_core.knowledge.sources.pubmed import search_pubmed
        try:
            articles = await search_pubmed(query, max_results=5)
            return [
                RAGChunk(
                    content=a.get("content", "")[:self.max_chars_per_chunk],
                    source="pubmed",
                    reference_id=f"PMID:{a.get('pmid', 'unknown')}",
                    metadata={"pmid": a.get("pmid")},
                )
                for a in articles if a.get("content")
            ]
        except Exception as e:
            logger.debug("[RAG/PubMed] %s", e)
            return []

    async def _fetch_opentargets(self, query: str) -> list[RAGChunk]:
        from vinci_core.knowledge.sources.opentargets import get_drug_disease_associations
        # Extract likely drug/target name from query (first capitalized word heuristic)
        tokens = query.split()
        candidate = next((t for t in tokens if t[0].isupper() and len(t) > 3), None)
        if not candidate:
            return []
        try:
            associations = await get_drug_disease_associations(candidate, limit=4)
            return [
                RAGChunk(
                    content=a.get("content", "")[:self.max_chars_per_chunk],
                    source="opentargets",
                    reference_id=a.get("disease_id", "OT"),
                    metadata={"drug": a.get("drug"), "disease": a.get("disease_name"), "phase": a.get("clinical_phase")},
                )
                for a in associations if a.get("content")
            ]
        except Exception as e:
            logger.debug("[RAG/OpenTargets] %s", e)
            return []

    async def _fetch_clinicaltrials(self, query: str, country: Optional[str] = None) -> list[RAGChunk]:
        """Fetch from ClinicalTrials.gov API v2 (public, no key required)."""
        import httpx
        ct_country_map = {
            "brazil": "BR", "mexico": "MX", "colombia": "CO",
            "argentina": "AR", "chile": "CL",
        }
        params: dict = {
            "query.cond": query,
            "pageSize": 3,
            "format": "json",
            "fields": "NCTId,BriefTitle,BriefSummary,Phase,OverallStatus",
        }
        if country:
            params["query.locn"] = ct_country_map.get(country.lower(), country)
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get("https://clinicaltrials.gov/api/v2/studies", params=params)
                r.raise_for_status()
                studies = r.json().get("studies", [])
            chunks = []
            for s in studies:
                proto = s.get("protocolSection", {})
                id_mod = proto.get("identificationModule", {})
                desc_mod = proto.get("descriptionModule", {})
                chunks.append(RAGChunk(
                    content=f"{id_mod.get('briefTitle', '')}. {desc_mod.get('briefSummary', '')}"[:self.max_chars_per_chunk],
                    source="clinicaltrials",
                    reference_id=id_mod.get("nctId", "NCT"),
                    metadata={"phase": proto.get("designModule", {}).get("phases", [])},
                ))
            return chunks
        except Exception as e:
            logger.debug("[RAG/ClinicalTrials] %s", e)
            return []

    # ---------------------------------------------------------------------------
    # Scoring and assembly
    # ---------------------------------------------------------------------------

    def _expand_query(self, query: str) -> str:
        """Add MeSH synonyms to improve recall."""
        extra_terms: list[str] = []
        q_lower = query.lower()
        for term, synonyms in _MESH_SYNONYMS.items():
            if term in q_lower:
                extra_terms.extend(synonyms[:2])
        if extra_terms:
            return f"{query} {' '.join(extra_terms)}"
        return query

    def _score_chunks(self, chunks: list[RAGChunk], query: str) -> list[RAGChunk]:
        """
        TF-IDF inspired relevance scoring without external dependencies.
        In production: replace with sentence-transformer cosine similarity.
        """
        query_terms = set(re.findall(r"\b\w{4,}\b", query.lower()))

        for chunk in chunks:
            if not chunk.content:
                chunk.relevance_score = 0.0
                continue
            content_lower = chunk.content.lower()
            words = re.findall(r"\b\w{4,}\b", content_lower)
            if not words:
                chunk.relevance_score = 0.0
                continue
            total = len(words)
            term_hits = sum(1 for w in words if w in query_terms)
            tf = term_hits / total
            # Boost PubMed peer-reviewed content slightly
            source_weight = 1.2 if chunk.source == "pubmed" else 1.0
            # IDF approximation: reward rarer query term matches
            unique_hits = len(query_terms & set(words))
            idf_approx = math.log(1 + unique_hits)
            chunk.relevance_score = round(tf * idf_approx * source_weight, 4)

        return sorted(chunks, key=lambda c: c.relevance_score, reverse=True)

    def _assemble_context(self, chunks: list[RAGChunk]) -> str:
        """Pack top chunks into context budget."""
        parts: list[str] = []
        used_chars = 0
        for chunk in chunks:
            if used_chars >= self.context_budget:
                break
            snippet = chunk.content[:self.max_chars_per_chunk]
            block = f"[{chunk.source.upper()} | {chunk.reference_id}]\n{snippet}"
            parts.append(block)
            used_chars += len(block)
        return "\n\n".join(parts)


rag_pipeline = RAGPipeline()
