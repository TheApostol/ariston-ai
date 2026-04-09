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
"""

import time
import uuid
import logging
from vinci_core.schemas import AIResponse
from vinci_core.routing.model_router import ModelRouter
from vinci_core.safety.guardrails import SafetyGuardrails, check_safety
from vinci_core.context.builder import build_context
from vinci_core.agent.classifier import classifier
from vinci_core.agent.patient_agent import patient_agent
from vinci_core.agent.genomics_agent import pharmacogenomics_agent
from vinci_core.evaluation.benchmark_logger import benchmark_logger
from app.services.audit_ledger import AristonAuditLedger

logger = logging.getLogger("ariston.engine")


class Engine:
    def __init__(self):
        self.router = ModelRouter()

    async def run(
        self,
        prompt: str,
        model: str = None,
        layer: str = None,
        context: dict = None,
        use_rag: bool = True,
        patient_id: str = None,
    ) -> AIResponse:
        request_id = str(uuid.uuid4())
        context = context or {}
        t_start = time.monotonic()

        try:
            # 1. Input validation
            valid, prompt, input_meta = SafetyGuardrails.validate_input(prompt)
            if not valid:
                logger.warning(
                    '{"event": "input_blocked", "request_id": "%s", "reason": "%s"}',
                    request_id, input_meta.get("safety_flag"),
                )
                return AIResponse(
                    model="vinci-safety",
                    content=prompt,
                    usage=None,
                    metadata={
                        "error": True,
                        "safety": input_meta,
                        "request_id": request_id,
                        "latency_ms": 0,
                    },
                )

            # 2. Auto-classify layer if not provided
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
                pgx_text = await pharmacogenomics_agent.format_for_context(drug_name)
                context["pharmacogenomics"] = pgx_text

            # 5. RAG enrichment
            enriched_context = await build_context(
                prompt=prompt,
                context=context,
                layer=layer,
                use_rag=use_rag,
            )

            # 6. Model execution
            result = await self.router.run(
                prompt=prompt,
                model=model,
                layer=layer,
                context=enriched_context,
                request_id=request_id,
            )

            # 7. Output safety validation
            content = result.get("content", "")
            safe, content, output_meta = SafetyGuardrails.validate_output(content)

            # 8. Build response metadata
            safety_meta = check_safety(content)
            safety_meta.update(output_meta)

            raw_usage = result.get("usage") or {}
            usage = {
                "prompt_tokens": raw_usage.get("prompt_tokens", 0),
                "completion_tokens": raw_usage.get("completion_tokens", 0),
                "total_tokens": raw_usage.get("total_tokens", 0),
            }

            latency_ms = round((time.monotonic() - t_start) * 1000)
            metadata = result.get("metadata", {}) or {}
            metadata.update({
                "safety": safety_meta,
                "layer": layer,
                "request_id": request_id,
                "latency_ms": latency_ms,
                "rag_used": use_rag,
                "consensus": metadata.get("consensus", False),
                "fallback_used": metadata.get("fallback_used", False),
            })

            logger.info(
                '{"event": "request_complete", "request_id": "%s", "model": "%s", '
                '"layer": "%s", "latency_ms": %d, "fallback_used": %s, "safety_flag": "%s"}',
                request_id,
                result.get("model", "unknown"),
                layer,
                latency_ms,
                str(metadata.get("fallback_used", False)).lower(),
                safety_meta.get("flag", "SAFE"),
            )

            response = AIResponse(
                model=result.get("model", "unknown"),
                content=content,
                usage=usage,
                metadata=metadata,
            )

            # 9. Benchmark + audit
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

            return response

        except Exception as exc:
            latency_ms = round((time.monotonic() - t_start) * 1000)
            logger.error(
                '{"event": "request_error", "request_id": "%s", "error": "%s", "latency_ms": %d}',
                request_id, str(exc), latency_ms,
            )
            return AIResponse(
                model="vinci",
                content="An internal error occurred. Please try again.",
                usage=None,
                metadata={"error": True, "request_id": request_id, "latency_ms": latency_ms},
            )


engine = Engine()
