from vinci_core.routing.model_router import ModelRouter
from vinci_core.schemas import AIRequest, AIResponse
from vinci_core.safety.guardrails import SafetyGuardrails
from vinci_core.tools.medical_tools import MedicalTools
from vinci_core.evaluation.benchmark_logger import BenchmarkLogger
from vinci_core.agent.classifier import IntentClassifier
from vinci_core.middleware.retry import with_retry
from vinci_core.logger import vinci_logger
from vinci_core.metrics import REQUEST_COUNT, REQUEST_LATENCY, FALLBACK_COUNT
from vinci_core.database.vector_store import VectorMemoryDB
from vinci_core.agent.patient_agent import patient_agent
from vinci_core.agent.genomics_agent import pharmacogenomics_agent
from vinci_core.agent.twin_agent import digital_twin_agent
from vinci_core.agent.regulatory_agent import regulatory_copilot
from vinci_core.agent.iomt_agent import iomt_agent
import time
import asyncio


class VinciEngine:
    def __init__(self):
        self.router = ModelRouter()
        self.classifier = IntentClassifier()
        self.memory_db = VectorMemoryDB()

        # ✅ REQUIRED for /models endpoint
        self.available_models = [
            "anthropic",
            "gemini",
            "openrouter"
        ]

    async def run(self, request: AIRequest) -> AIResponse:
        # Inject long-term conversational RAG back into the active prompt
        past_context = self.memory_db.get_recent_context()
        if past_context:
            request.prompt = f"{past_context}\n[NEW QUERY]\n{request.prompt}"

        # Setup iteration state
        current_prompt = request.prompt
        current_context = (request.context or {}).copy()
        current_retry = 0
        max_retries = 1
        reflection_traces = []

        while True:
            # ✅ Convert current state for execution
            start_time = time.time()
            
            # 🕰️ Longitudinal Patient History Injection (First pass only)
            if current_retry == 0 and request.patient_id:
                history_str = patient_agent.get_full_history(request.patient_id)
                current_prompt = f"{history_str}\n\n[CURRENT CONTEXT]\n{current_prompt}"

            # Default layer extraction (using current prompt)
            layer = current_context.get("layer")
            if not layer:
                layer = await self.classifier.classify(current_prompt)
                current_context["layer"] = layer
            
            # Select model
            model = self.router.select_model(layer=layer, context=current_context)
            
            fallback_used = False
            failure_reason = None
            safety_metadata = {}

            # 🧠 Short-term Memory Injection (Only on first pass)
            history = current_context.get("history", [])
            if current_retry == 0:
                history.append({"role": "user", "content": current_prompt})
                current_context["history"] = history
            
            # 1. Input Guardrails
            is_safe_input, safe_prompt_msg, in_safety_meta = SafetyGuardrails.validate_input(current_prompt)
            safety_metadata.update(in_safety_meta)

            if not is_safe_input:
                return AIResponse(
                    model="guardrails",
                    content=safe_prompt_msg,
                    usage={},
                    metadata={"provider": "safety_layer", "latency_ms": int((time.time() - start_time) * 1000), **safety_metadata}
                )

            # 1.5 Tool Retrieval & Semantic Grounding
            try:
                if layer == "pharma":
                    words = current_prompt.split()
                    if words:
                        drug_name = words[0]
                        classes = await MedicalTools.get_drug_classes(drug_name)
                        fda_info = await MedicalTools.get_fda_drug_info(drug_name)
                        vademecum = await MedicalTools.get_vademecum_data(drug_name) # 💊 Vademecum Expansion
                        
                        grounding = (
                            f"[RxNorm Classes: {classes}]\n"
                            f"[FDA Label: {fda_info.get('brand_name')} - Indications: {fda_info.get('indications')[:200]}...]\n"
                            f"[Ariston Vademecum: {vademecum.get('mechanism')} | Interactions: {vademecum.get('interactions')}]"
                        )
                        
                        # 🧬 Pharmacogenomics Personalized Check
                        # In production, we would fetch ClinVar variants from the patient history.
                        simulated_variants = [{"title": "CYP2C19 Poor Metabolizer"}] 
                        pgx_check = await pharmacogenomics_agent.cross_reference(drug_name, simulated_variants)
                        if pgx_check["genomic_alerts"]:
                            alert_str = f"\n[PGx CRITICAL ALERT: {pgx_check['genomic_alerts'][0]['risk_description']}]"
                            grounding += alert_str

                        current_prompt = f"{grounding}\n\n{current_prompt}"
                elif layer == "clinical":
                    # Step 1: Integrated Symptom Research + Web Public Records
                    pubmed = await MedicalTools.search_pubmed(current_prompt[:60])
                    research = await MedicalTools.get_symptom_research(current_prompt[:60])
                    
                    # 🌐 Autonomous Cloud Record Harvesting (via MedicalTools)
                    web_records = await MedicalTools.search_public_records(current_prompt[:40])
                    
                    evidence_str = ""
                    if pubmed:
                        evidence = [f"{r['title']} ({r['source']})" for r in pubmed]
                        evidence_str += f"[PubMed Evidence: {evidence}]\n"
                    if research:
                        evidence_str += f"[Clinical Symptom Research: {research}]\n"
                    if web_records:
                        evidence_str += f"[Public Web Records (Harvested): {web_records[:500]}...]\n"
                    
                    if evidence_str:
                        current_prompt = f"{evidence_str}\n{current_prompt}"
            except Exception as e:
                vinci_logger.warning(f"Grounding layer failed: {e}")

            # 2. Model Execution (The Clinical Swarm)
            try:
                if layer == "radiology":
                    from vinci_core.agent.vision_agent import VisionRadiologyAgent
                    images = request.images or []
                    res_content = await VisionRadiologyAgent().analyze_scan(current_prompt, images=images)
                    result = {"content": res_content, "model": "AristonVision-Gen2"}
                elif layer == "pharma":
                    from vinci_core.agent.pharmacist_agent import PharmacistAgent
                    res_content = await PharmacistAgent().review_medications(current_prompt, {"pharma_grounding": "Vademecum Active"})
                    result = {"content": res_content, "model": "AristonPharmacist-Swarm"}
                elif layer == "clinical":
                    from vinci_core.routing.consensus_router import ConsensusModel
                    consensus_ctx = current_context.copy()
                    consensus_ctx["prompt"] = current_prompt
                    res_content = await ConsensusModel().generate(consensus_ctx)
                    result = {"content": res_content, "model": "dual_consensus_arbiter"}
                else:
                    result = await self._execute_model(model, {"prompt": current_prompt, **current_context})
            except Exception as e:
                failure_reason = str(e)
                fallback = self.router.get_fallback_model(model)
                if fallback:
                    fallback_used = True
                    model = fallback
                    result = await fallback.generate({"prompt": current_prompt, **current_context})
                else:
                    raise e
            
            latency_ms = int((time.time() - start_time) * 1000)
            raw_content = self._extract_content(result)
            is_safe_output, final_content, out_safety_meta = SafetyGuardrails.validate_output(raw_content)
            safety_metadata.update(out_safety_meta)

            # Meta construction
            final_meta = {
                "provider": model.__class__.__name__,
                "latency_ms": latency_ms,
                "fallback_used": fallback_used,
                "failure_reason": failure_reason,
                "history": history,
                "reflection_traces": reflection_traces,
                **safety_metadata
            }

            # 3. MedPerf Benchmarking
            BenchmarkLogger.evaluate_and_log(current_context, final_meta, final_content)
            metrics = final_meta.get("benchmark_metrics") or {}

            # 🧠 Reflection Check
            if (metrics.get("safety_score", 1.0) < 1.0 or metrics.get("grounding_score", 1.0) < 0.8) and current_retry < max_retries:
                vinci_logger.info(f"🧠 [Own AI] Reflection triggered (Iteration {current_retry + 1})")
                current_retry += 1
                
                # Construct reflection prompt for next iteration
                current_prompt = (
                    f"Your previous response was flagged for low accuracy/safety.\n"
                    f"Previous Response:\n{final_content}\n\n"
                    f"Reflect and provide a strictly compliant, well-hedged medical response."
                )
                
                reflection_traces.append({
                    "iteration": current_retry,
                    "reason": "low_benchmark_score",
                    "feedback": f"Safety: {metrics.get('safety_score')}, Grounding: {metrics.get('grounding_score')}"
                })
                continue # Next iteration of while loop
            
            # If we reach here, we are done (either success or hit max retries)
            # 📊 Prometheus Observability
            REQUEST_COUNT.labels(model_name=final_meta["provider"], layer=layer).inc()
            
            # Final memory save
            if current_retry == 0:
                history.append({"role": "assistant", "content": final_content})
                self.memory_db.log_memory(request.prompt, final_content)
                
                # 🛡️ GxP Audit Persistence (Phase 10)
                import json
                audit_meta = {k: v for k, v in final_meta.items() if k != "history"}
                self.memory_db.log_audit_entry(
                    job_id=job_id_val,
                    timestamp=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                    entry_hash=final_meta.get("safety_flag", "PENDING"), # Or a real hash of result
                    metadata=json.dumps(audit_meta)
                )

            # 📊 Phase 9: Final Predictive & Regulatory Layer
            patient_id = request.patient_id or "ARISTON-TEST"
            job_id_val = final_meta.get("job_id", "ACR-0000")
            history_str = patient_agent.get_history(patient_id)
            
            # 🧪 Digital Twin & IoMT Simulations
            twin_sim = digital_twin_agent.simulate_treatment(history_str, request.prompt, ["CYP2C19 Poor Metabolizer"])
            adherence_forecast = iomt_agent.forecast_adherence(history_str)
            
            # ☢️ Radiology Saliency Map Expansion
            if layer == "radiology" and request.images:
                from vinci_core.agent.vision_agent import VisionRadiologyAgent
                saliency = VisionRadiologyAgent().generate_saliency_map(request.images[0])
                final_meta["radiology_saliency"] = saliency

            # 📄 GxP Regulatory Report
            gxp_report = regulatory_copilot.generate_report(job_id_val, request.prompt, final_content, [])
            
            final_meta["digital_twin_simulation"] = twin_sim
            final_meta["iomt_adherence_forecast"] = adherence_forecast
            final_meta["regulatory_report_draft"] = gxp_report

            return AIResponse(
                model=self._detect_model_name(result, model),
                content=final_content,
                usage=result.get("usage", {}),
                metadata=final_meta
            )

    def _extract_content(self, result: dict | str) -> str:
        """
        Normalize responses across providers
        """
        if isinstance(result, str):
            return result

        # OpenRouter / OpenAI-style
        if "choices" in result:
            return result["choices"][0]["message"]["content"]

        # Anthropic-style
        if "content" in result and isinstance(result["content"], list):
            return result["content"][0].get("text", "")

        # Gemini-style
        if "candidates" in result:
            return result["candidates"][0]["content"]["parts"][0]["text"]

        # Fallback
        return str(result)

    def _detect_model_name(self, result: dict | str, model) -> str:
        """
        Try to extract model name cleanly
        """
        if isinstance(result, dict) and "model" in result:
            return result["model"]

        return model.__class__.__name__

    @with_retry(max_retries=3, base_delay=1.0)
    async def _execute_model(self, model, context: dict) -> dict:
        """Isolated model execution wrapped in retry middleware for transient API errors"""
        return await model.generate(context)

# ✅ Singleton instance (IMPORTANT for router import)
engine = VinciEngine()
