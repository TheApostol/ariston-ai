from vinci_core.routing.model_router import ModelRouter
from vinci_core.schemas import AIRequest, AIResponse
from vinci_core.safety.guardrails import SafetyGuardrails
from vinci_core.tools.medical_tools import MedicalTools
from vinci_core.evaluation.benchmark_logger import BenchmarkLogger
from vinci_core.agent.classifier import IntentClassifier
import time
import time


class VinciEngine:
    def __init__(self):
        self.router = ModelRouter()
        self.classifier = IntentClassifier()

        # ✅ REQUIRED for /models endpoint
        self.available_models = [
            "anthropic",
            "gemini",
            "openrouter"
        ]

    async def run(self, request: AIRequest) -> AIResponse:
        # ✅ Convert Pydantic → dict (fixes serialization bug)
        context = request.model_dump()

        # Default layer extraction (handle Pydantic nesting)
        user_context = context.get("context") or {}
        layer = user_context.get("layer")

        # 🧠 "Own AI" Autonomous Routing
        prompt = context.get("prompt", "")
        if not layer:
            layer = await self.classifier.classify(prompt)
            # Inject it so plugins know
            context["layer"] = layer
        else:
            context["layer"] = layer # standardize flat access for downstream

        # Select model
        model = self.router.select_model(layer=layer, context=context)
        start_time = time.time()
        
        fallback_used = False
        failure_reason = None
        safety_metadata = {}

        # 1. Input Guardrails
        prompt = context.get("prompt", "")
        is_safe_input, safe_prompt_msg, in_safety_meta = SafetyGuardrails.validate_input(prompt)
        safety_metadata.update(in_safety_meta)

        if not is_safe_input:
            latency_ms = int((time.time() - start_time) * 1000)
            return AIResponse(
                model="guardrails",
                content=safe_prompt_msg,
                usage={},
                metadata={
                    "provider": "safety_layer",
                    "latency_ms": latency_ms,
                    "fallback_used": False,
                    "failure_reason": "input_validation_failed",
                    **safety_metadata
                }
            )

        # 1.5 Tool Retrieval (Copilot)
        try:
            if layer == "pharma":
                # Naive drug extraction from prompt
                words = prompt.split()
                if words:
                    classes = await MedicalTools.get_drug_classes(words[0])
                    if classes and classes[0] != "Unknown drug or no RxCUI found.":
                        context["prompt"] = f"[RxNorm Classes: {classes}]\n\n{context['prompt']}"
            elif layer == "clinical":
                pubmed_results = await MedicalTools.search_pubmed(prompt[:60])
                if pubmed_results:
                    # Format evidence compactly
                    evidence = [f"{r['title']} ({r['source']})" for r in pubmed_results]
                    context["prompt"] = f"[PubMed Evidence: {evidence}]\n\n{context['prompt']}"
        except Exception as e:
            print(f"Tool retrieval failed softly: {e}")

        try:
            result = await model.generate(context)

        except Exception as e:
            failure_reason = str(e)
            print(f"⚠️ Primary model failed: {failure_reason}")

            # 🔁 Fallback logic
            fallback = self.router.get_fallback_model(model)

            if fallback:
                fallback_used = True
                print(f"🔁 Using fallback: {fallback.__class__.__name__}")
                model = fallback  # Update model reference for metadata
                result = await fallback.generate(context)
            else:
                raise e
        
        latency_ms = int((time.time() - start_time) * 1000)

        # 2. Output Guardrails
        raw_content = self._extract_content(result)
        is_safe_output, final_content, out_safety_meta = SafetyGuardrails.validate_output(raw_content)
        safety_metadata.update(out_safety_meta)

        # Normalize response metadata
        final_meta = {
            "provider": model.__class__.__name__,
            "latency_ms": latency_ms,
            "fallback_used": fallback_used,
            "failure_reason": failure_reason,
            **safety_metadata
        }

        # 3. MedPerf Benchmarking
        BenchmarkLogger.evaluate_and_log(context, final_meta, final_content)

        return AIResponse(
            model=self._detect_model_name(result, model),
            content=final_content,
            usage=result.get("usage", {}),
            metadata=final_meta
        )

    def _extract_content(self, result: dict) -> str:
        """
        Normalize responses across providers
        """

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

    def _detect_model_name(self, result: dict, model) -> str:
        """
        Try to extract model name cleanly
        """
        if "model" in result:
            return result["model"]

        return model.__class__.__name__


# ✅ Singleton instance (IMPORTANT for router import)
engine = VinciEngine()
