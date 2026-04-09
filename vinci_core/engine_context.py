"""
Shared request pre-processing pipeline used by both engine.py (batch) and
engine_stream.py (streaming).

Steps:
  1. Input safety validation
  2. Intent classification (auto-detect layer)
  3. Patient history injection (if patient_id provided)
  4. PGx grounding (if drug_name in context)
  5. RAG context enrichment
"""

import logging
from typing import Optional, Tuple

from vinci_core.safety.guardrails import SafetyGuardrails
from vinci_core.agent.classifier import classifier
from vinci_core.agent.patient_agent import patient_agent
from vinci_core.agent.genomics_agent import pharmacogenomics_agent
from vinci_core.context.builder import build_context

logger = logging.getLogger("ariston.engine")


async def build_request_context(
    prompt: str,
    layer: Optional[str],
    context: dict,
    use_rag: bool,
    patient_id: Optional[str],
    request_id: str,
) -> Tuple[bool, str, Optional[str], dict]:
    """
    Run the shared pre-processing pipeline.

    Returns:
        (valid, prompt, layer, enriched_context)

        If valid is False, prompt contains the safety rejection message and
        the caller should return immediately without invoking a model.
    """
    # 1. Input validation
    valid, prompt, input_meta = SafetyGuardrails.validate_input(prompt)
    if not valid:
        logger.warning(
            '{"event":"input_blocked","request_id":"%s"}',
            request_id,
        )
        return False, prompt, layer, {}

    # 2. Auto-classify layer
    if not layer:
        layer = await classifier.classify(prompt)

    # 3. Patient history injection
    if patient_id:
        history = patient_agent.get_full_history(patient_id)
        if history:
            context["patient_history"] = history

    # 4. PGx grounding
    drug_name = context.get("drug_name") or context.get("drug")
    if drug_name:
        context["pharmacogenomics"] = await pharmacogenomics_agent.format_for_context(drug_name)

    # 5. RAG enrichment
    enriched_context = await build_context(
        prompt=prompt,
        context=context,
        layer=layer,
        use_rag=use_rag,
    )

    return True, prompt, layer, enriched_context
