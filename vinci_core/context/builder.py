"""
Context builder — enriches every engine request with RAG-retrieved knowledge.

Routes to the new RAGPipeline (multi-source: PubMed + OpenTargets + ClinicalTrials)
which replaced the stub retriever. Falls back to legacy retrieve() on error.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger("ariston.context")


async def build_context(
    prompt: str,
    context: Optional[dict],
    layer: str,
    use_rag: bool = True,
) -> dict:
    enriched: dict = {"prompt": prompt, "layer": layer}

    if context:
        enriched.update(context)

    if not use_rag:
        return enriched

    # --- RAG enrichment via production pipeline ---
    try:
        from vinci_core.rag.pipeline import rag_pipeline

        use_ct = layer in ("clinical", "pharma", "latam")
        use_ot = layer in ("pharma", "data")

        rag_result = await rag_pipeline.retrieve(
            query=prompt,
            layer=layer,
            use_pubmed=True,
            use_opentargets=use_ot,
            use_clinicaltrials=use_ct,
            country_filter=context.get("country") if context else None,
        )

        if rag_result.context_text:
            enriched["retrieved_knowledge"] = rag_result.context_text
            enriched["rag_sources"] = rag_result.sources_cited
            enriched["rag_chunks_used"] = rag_result.total_chunks_used
            enriched["rag_query_expanded"] = rag_result.query_expanded

    except Exception as e:
        logger.warning("[Context] RAG pipeline error, falling back to legacy retriever: %s", e)
        # Graceful fallback to legacy retriever
        try:
            from vinci_core.knowledge.retriever import retrieve, format_context
            chunks = await retrieve(prompt, layer=layer, max_results=5)
            rag_text = format_context(chunks)
            if rag_text:
                enriched["retrieved_knowledge"] = rag_text
        except Exception as e2:
            logger.warning("[Context] Legacy retriever also failed: %s", e2)

    return enriched
