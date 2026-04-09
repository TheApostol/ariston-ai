"""
Context builder — enriches every request with RAG-retrieved knowledge.
"""

from typing import Optional
from vinci_core.knowledge.retriever import retrieve, format_context


async def build_context(
    prompt: str,
    context: Optional[dict],
    layer: str,
    use_rag: bool = True,
) -> dict:
    enriched = {
        "prompt": prompt,
        "layer": layer,
    }

    if context:
        enriched.update(context)

    if use_rag:
        chunks = await retrieve(prompt, layer=layer, max_results=5)
        rag_text = format_context(chunks)
        if rag_text:
            enriched["retrieved_knowledge"] = rag_text

    return enriched
