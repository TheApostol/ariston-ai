from vinci_core.routing.model_router import ModelRouter
from vinci_core.schemas import AIRequest, AIResponse
import time


class VinciEngine:
    def __init__(self):
        self.router = ModelRouter()

        # ✅ REQUIRED for /models endpoint
        self.available_models = [
            "anthropic",
            "gemini",
            "openrouter"
        ]

    async def run(self, request: AIRequest) -> AIResponse:
        # ✅ Convert Pydantic → dict (fixes serialization bug)
        context = request.model_dump()

        # Default layer
        layer = context.get("layer", "general")

        # Select model
        model = self.router.select_model(layer=layer, context=context)
        start_time = time.time()
        
        fallback_used = False
        failure_reason = None

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

        # Normalize response
        return AIResponse(
            model=self._detect_model_name(result, model),
            content=self._extract_content(result),
            usage=result.get("usage", {}),
            metadata={
                "provider": model.__class__.__name__,
                "latency_ms": latency_ms,
                "fallback_used": fallback_used,
                "failure_reason": failure_reason
            }
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
