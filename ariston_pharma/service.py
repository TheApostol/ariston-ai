"""
Ariston Pharma — regulatory and pharmaceutical AI layer.
"""

from typing import Dict, Any, Optional
from vinci_core.engine import engine


def build_pharma_context(
    document: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:

    context: Dict[str, Any] = {}

    if document:
        context["document"] = document

    if metadata:
        context["metadata"] = metadata

    return context


async def pharma_query(
    prompt: str,
    document: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:

    context = build_pharma_context(document, metadata)

    return await engine.run(
        prompt=prompt,
        context=context,
        layer="pharma"
    )
