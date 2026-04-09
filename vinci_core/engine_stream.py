"""
Streaming engine — same pipeline as engine.py but yields tokens via async generator.
Used by the /stream endpoint for real-time response delivery.

The shared pre-processing pipeline (input safety, classification, patient
history, PGx, RAG) is handled by vinci_core/engine_context.py so the two
engines stay in sync without code duplication.
"""

import uuid
import logging
from typing import AsyncGenerator, Optional

import anthropic
from config import settings
from vinci_core.engine_context import build_request_context
from vinci_core.safety.guardrails import SafetyGuardrails
from vinci_core.layers.base_layer import BaseLayer
from vinci_core.layers.pharma_layer import PharmaLayer
from vinci_core.layers.clinical_layer import ClinicalLayer
from vinci_core.layers.data_layer import DataLayer
from app.services.audit_ledger import AristonAuditLedger

logger = logging.getLogger("ariston.engine.stream")

_LAYERS = {
    "base": BaseLayer(), "pharma": PharmaLayer(),
    "clinical": ClinicalLayer(), "data": DataLayer(),
    "radiology": ClinicalLayer(), "general": BaseLayer(),
}


async def stream_response(
    prompt: str,
    layer: Optional[str] = None,
    context: Optional[dict] = None,
    use_rag: bool = True,
    patient_id: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """
    Async generator — yields text chunks as they arrive from Claude.
    Caller sends each chunk over WebSocket or SSE.

    Output safety (definitive-diagnosis blocking) is enforced on the fully
    assembled response after streaming completes; the stream is replaced with
    the safety fallback message if a violation is detected.
    """
    job_id = str(uuid.uuid4())
    context = context or {}

    # 1–5. Shared pre-processing pipeline
    valid, prompt, layer, enriched = await build_request_context(
        prompt=prompt,
        layer=layer,
        context=context,
        use_rag=use_rag,
        patient_id=patient_id,
        request_id=job_id,
    )
    if not valid:
        yield prompt
        return

    # Build messages
    layer_obj = _LAYERS.get(layer, _LAYERS["base"])
    messages = layer_obj.build_messages(prompt, enriched)
    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    user_msgs = [m for m in messages if m["role"] != "system"]

    client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
    full_content: list[str] = []

    async with client.messages.stream(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        system="\n\n".join(system_parts) or "You are Ariston AI.",
        messages=user_msgs,
    ) as stream:
        async for text in stream.text_stream:
            full_content.append(text)
            yield text

    # 6. Output safety — enforce on the fully assembled content.
    #    If a violation is detected the stream has already been delivered, so
    #    we yield the replacement message as a final corrective chunk and audit
    #    the real content (never the raw chunk mid-stream).
    full = "".join(full_content)
    _safe, safe_content, _meta = SafetyGuardrails.validate_output(full)
    if not _safe:
        logger.warning(
            '{"event":"stream_output_blocked","request_id":"%s"}',
            job_id,
        )
        # Signal the client to replace the streamed content
        yield f"\n\n[SAFETY_OVERRIDE]{safe_content}"

    # 7. Post-stream audit (non-blocking)
    AristonAuditLedger.log_decision(
        job_id=job_id, prompt=prompt, result=safe_content,
        metadata={"layer": layer, "streaming": True}
    )

