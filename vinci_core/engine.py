"""
Vinci Engine — main orchestration loop.

Full pipeline per request:
  1. Input safety validation
  2. Intent classification (auto-detect layer)
  3. Patient history injection (if patient_id provided)
  4. RAG context enrichment (PubMed, FDA, ClinicalTrials)
  5. PGx grounding (if drug_name in context)
  6. Model routing + execution (consensus for clinical)
  7. Output safety validation + definitive diagnosis blocking
  8. MedPerf benchmarking
  9. GxP audit ledger entry

Observability: every request carries a request_id and emits structured
JSON-compatible log lines (layer, model, latency_ms, fallback_used, safety_flag).
"""

import logging
import time
import uuid
from typing import Optional

from vinci_core.schemas import AIResponse
from vinci_core.routing.model_router import ModelRouter
from vinci_core.safety.guardrails import SafetyGuardrails, check_safety
from vinci_core.engine_context import build_request_context
from vinci_core.evaluation.benchmark_logger import benchmark_logger, _SAFE_LAYER_NAMES
from app.services.audit_ledger import AristonAuditLedger

logger = logging.getLogger("ariston.engine")


class Engine:
    def __init__(self):
        self.router = ModelRouter()

    async def run(
        self,
        prompt: str,
        model: Optional[str] = None,
        layer: Optional[str] = None,
        context: Optional[dict] = None,
        use_rag: bool = True,
        patient_id: Optional[str] = None,
    ) -> AIResponse:
        request_id = str(uuid.uuid4())
        context = context or {}
        start_time = time.monotonic()

        logger.info(
            '{"event":"engine_start","request_id":"%s","layer":"%s"}',
            request_id, layer or "auto",
        )

        try:
            # 1–5. Input validation, classification, patient history, PGx, RAG
            valid, prompt, layer, enriched_context = await build_request_context(
                prompt=prompt,
                layer=layer,
                context=context,
                use_rag=use_rag,
                patient_id=patient_id,
                request_id=request_id,
            )
            if not valid:
                return AIResponse(
                    model="vinci-safety",
                    content=prompt,
                    usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    metadata={"error": True, "request_id": request_id},
                )

            # 6. Model execution
            result = await self.router.run(
                prompt=prompt,
                model=model,
                layer=layer,
                context=enriched_context,
            )

            # 7. Output safety validation
            content = result.get("content", "")
            _safe, content, output_meta = SafetyGuardrails.validate_output(content)

            # 8. Build response metadata
            safety_meta = check_safety(content)
            safety_meta.update(output_meta)

            raw_usage = result.get("usage") or {}
            usage = {
                "prompt_tokens": raw_usage.get("prompt_tokens", 0) or 0,
                "completion_tokens": raw_usage.get("completion_tokens", 0) or 0,
                "total_tokens": raw_usage.get("total_tokens", 0) or 0,
            }

            latency_ms = round((time.monotonic() - start_time) * 1000)
            result_meta = result.get("metadata") or {}
            metadata = {
                **result_meta,
                "safety": safety_meta,
                "layer": layer,
                "job_id": request_id,
                "request_id": request_id,
                "rag_used": use_rag,
                "consensus": result_meta.get("consensus", False),
                "latency_ms": latency_ms,
                "provider": result_meta.get("provider", "unknown"),
            }

            response = AIResponse(
                model=result.get("model", "unknown"),
                content=content,
                usage=usage,
                metadata=metadata,
            )

            logger.info(
                '{"event":"engine_complete","request_id":"%s","model":"%s","layer":"%s",'
                '"latency_ms":%d,"safety_flag":"%s","rag_used":%s}',
                request_id, response.model,
                layer if layer in _SAFE_LAYER_NAMES else "unknown",
                latency_ms,
                safety_meta.get("flag", "SAFE"),
                str(use_rag).lower(),
            )

            # 8. Benchmark + audit (non-blocking; errors must not fail the request)
            try:
                benchmark_logger.evaluate_and_log(
                    prompt=prompt,
                    response_content=content,
                    response_metadata=metadata,
                    layer=layer,
                )
                AristonAuditLedger.log_decision(
                    job_id=request_id,
                    prompt=prompt,
                    result=content,
                    metadata=metadata,
                )
            except Exception as audit_exc:
                logger.warning(
                    '{"event":"audit_error","request_id":"%s","error":"%s"}',
                    request_id, type(audit_exc).__name__,
                )

            return response

        except Exception as e:
            latency_ms = round((time.monotonic() - start_time) * 1000)
            logger.error(
                '{"event":"engine_error","request_id":"%s","error_type":"%s","latency_ms":%d}',
                request_id, type(e).__name__, latency_ms,
            )
            # Never expose raw exception strings or stack traces to the API consumer
            return AIResponse(
                model="vinci",
                content=(
                    "An unexpected error occurred while processing your request. "
                    "Please try again or contact support if the issue persists."
                ),
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                metadata={
                    "error": True,
                    "error_type": type(e).__name__,
                    "request_id": request_id,
                    "latency_ms": latency_ms,
                },
            )


engine = Engine()

