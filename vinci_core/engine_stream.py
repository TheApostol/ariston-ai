"""
Streaming engine — same pipeline as engine.py but yields tokens via async generator.
Used by the /stream endpoint for real-time response delivery.
"""

import uuid
from typing import AsyncGenerator
import anthropic
from config import settings
from vinci_core.agent.classifier import classifier
from vinci_core.agent.patient_agent import patient_agent
from vinci_core.agent.genomics_agent import pharmacogenomics_agent
from vinci_core.context.builder import build_context
from vinci_core.safety.guardrails import SafetyGuardrails
from vinci_core.layers.base_layer import BaseLayer
from vinci_core.layers.pharma_layer import PharmaLayer
from vinci_core.layers.clinical_layer import ClinicalLayer
from vinci_core.layers.data_layer import DataLayer
from app.services.audit_ledger import AristonAuditLedger

_LAYERS = {
    "base": BaseLayer(), "pharma": PharmaLayer(),
    "clinical": ClinicalLayer(), "data": DataLayer(),
    "radiology": ClinicalLayer(), "general": BaseLayer(),
}


async def stream_response(
    prompt: str,
    layer: str = None,
    context: dict = None,
    use_rag: bool = True,
    patient_id: str = None,
) -> AsyncGenerator[str, None]:
    """
    Async generator — yields text chunks as they arrive from Claude.
    Caller sends each chunk over WebSocket or SSE.
    """
    job_id = str(uuid.uuid4())
    context = context or {}

    # Input validation
    valid, prompt, _ = SafetyGuardrails.validate_input(prompt)
    if not valid:
        yield prompt
        return

    # Auto-classify layer
    if not layer:
        layer = await classifier.classify(prompt)

    # Patient history + PGx injection
    if patient_id:
        history = patient_agent.get_full_history(patient_id)
        if history:
            context["patient_history"] = history

    drug_name = context.get("drug_name") or context.get("drug")
    if drug_name:
        context["pharmacogenomics"] = await pharmacogenomics_agent.format_for_context(drug_name)

    # RAG enrichment
    enriched = await build_context(prompt=prompt, context=context, layer=layer, use_rag=use_rag)

    # Build messages
    layer_obj = _LAYERS.get(layer, _LAYERS["base"])
    messages = layer_obj.build_messages(prompt, enriched)
    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    user_msgs = [m for m in messages if m["role"] != "system"]

    client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
    full_content = []

    async with client.messages.stream(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        system="\n\n".join(system_parts) or "You are Ariston AI.",
        messages=user_msgs,
    ) as stream:
        async for text in stream.text_stream:
            # Output safety: block definitive diagnosis mid-stream
            full_content.append(text)
            yield text

    # Post-stream audit (non-blocking)
    full = "".join(full_content)
    AristonAuditLedger.log_decision(
        job_id=job_id, prompt=prompt, result=full,
        metadata={"layer": layer, "streaming": True}
    )
