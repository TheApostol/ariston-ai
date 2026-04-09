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

import uuid
import time
from vinci_core.schemas import AIResponse
from vinci_core.routing.model_router import ModelRouter
from vinci_core.safety.guardrails import SafetyGuardrails, check_safety
from vinci_core.context.builder import build_context
from vinci_core.agent.classifier import classifier
from vinci_core.agent.patient_agent import patient_agent
from vinci_core.agent.genomics_agent import pharmacogenomics_agent
from vinci_core.evaluation.benchmark_logger import benchmark_logger
from vinci_core.observability.structured_logger import obs_logger, RequestTrace
from app.services.audit_ledger import AristonAuditLedger


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
        job_id = str(uuid.uuid4())
        t_start = time.perf_counter()
        context = context or {}

        try:
            # 1. Input validation
            valid, prompt, input_meta = SafetyGuardrails.validate_input(prompt)
            if not valid:
                return AIResponse(
                    model="vinci-safety",
                    content=prompt,
                    usage=None,
                    metadata={"error": True, "safety": input_meta},
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
                request_id=job_id,
            )

            # 7. Output safety validation
            content = result.get("content", "")
            safe, content, output_meta = SafetyGuardrails.validate_output(content)

            # 8. Build response metadata
            safety_meta = check_safety(content, layer=layer)
            safety_meta.update(output_meta)

            raw_usage = result.get("usage") or {}
            usage = {
                "prompt_tokens": raw_usage.get("prompt_tokens", 0),
                "completion_tokens": raw_usage.get("completion_tokens", 0),
                "total_tokens": raw_usage.get("total_tokens", 0),
            }

            metadata = result.get("metadata", {}) or {}
            metadata.update({
                "safety": safety_meta,
                "layer": layer,
                "job_id": job_id,
                "rag_used": use_rag,
                "consensus": metadata.get("consensus", False),
            })

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
                job_id=job_id,
                prompt=prompt,
                result=content,
                metadata=metadata,
            )

            # 10. Structured observability trace
            latency_ms = round((time.perf_counter() - t_start) * 1000, 2)
            trace = RequestTrace(
                request_id=job_id,
                layer=layer,
                model_requested=model,
                model_used=response.model,
                fallback_used=metadata.get("fallback_used", False),
                fallback_reason=metadata.get("fallback_reason"),
                latency_ms=latency_ms,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
                safety_flag=safety_meta.get("flag", "SAFE"),
                rag_used=use_rag,
                consensus=metadata.get("consensus", False),
            )
            obs_logger.emit(trace)
            metadata["latency_ms"] = latency_ms

            return response

        except Exception as e:
            latency_ms = round((time.perf_counter() - t_start) * 1000, 2)
            trace = RequestTrace(
                request_id=job_id,
                layer=layer or "unknown",
                latency_ms=latency_ms,
                safety_flag="ERROR",
                provider_errors=[str(e)],
            )
            obs_logger.emit(trace)
            return AIResponse(
                model="vinci",
                content=f"Internal error: {str(e)}",
                usage=None,
                metadata={"error": True, "job_id": job_id, "latency_ms": latency_ms},
            )


engine = Engine()
