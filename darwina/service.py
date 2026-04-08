"""
Darwina — data intelligence and predictive AI layer.
"""

from typing import Dict, Any, Optional
from vinci_core.engine import engine


def build_data_context(
    dataset: Optional[Dict[str, Any]] = None,
    signals: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:

    context: Dict[str, Any] = {}

    if dataset:
        context["dataset"] = dataset

    if signals:
        context["signals"] = signals

    return context


async def data_query(
    prompt: str,
    dataset: Optional[Dict[str, Any]] = None,
    signals: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:

    context = build_data_context(dataset, signals)

    return await engine.run(
        prompt=prompt,
        context=context,
        layer="data"
    )
