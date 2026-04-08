import asyncio
from vinci_core.models.anthropic_model import AnthropicModel
from vinci_core.models.gemini_model import GeminiModel
from vinci_core.models.openrouter_model import OpenRouterModel
from vinci_core.models.base import BaseModel

class ConsensusModel(BaseModel):
    def __init__(self):
        self.model_a = AnthropicModel()
        self.model_b = GeminiModel()
        self.arbiter = OpenRouterModel()
        self.name = "ConsensusModel"

    async def generate(self, context: dict) -> dict:
        # Run models in parallel
        try:
            res_a, res_b = await asyncio.gather(
                self.model_a.generate(context),
                self.model_b.generate(context)
            )
        except Exception as e:
            # Fallback to arbiter directly if sub-models fail to gather
            return await self.arbiter.generate(context)

        content_a = self._extract(res_a)
        content_b = self._extract(res_b)
        
        # Synthesize with arbiter
        synthesis_prompt = (
            f"Model A diagnostic thought:\n{content_a}\n\n"
            f"Model B diagnostic thought:\n{content_b}\n\n"
            f"User Prompt: {context.get('prompt')}\n\n"
            f"Analyze both thoughts. Synthesize them into a single, cohesive response "
            f"that is safe, well-hedged, and represents a multidisciplinary consensus."
        )
        
        synth_context = context.copy()
        synth_context["prompt"] = synthesis_prompt
        
        final_res = await self.arbiter.generate(synth_context)
        return final_res

    def _extract(self, result: dict) -> str:
        if "choices" in result:
            return result["choices"][0]["message"]["content"]
        if "content" in result and isinstance(result["content"], list):
            return result["content"][0].get("text", "")
        if "candidates" in result:
            return result["candidates"][0]["content"]["parts"][0]["text"]
        return str(result)
