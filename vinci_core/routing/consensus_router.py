import asyncio
from vinci_core.models.anthropic_model import AnthropicModel
from vinci_core.models.gemini_model import GeminiModel
from vinci_core.models.openrouter_model import OpenRouterModel
from vinci_core.models.base_model import BaseModel

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
            "You are the Ariston OS Clinical Consensus Arbiter. Your task is to synthesize contradictory or "
            "complementary diagnostic thoughts from multiple expert models into a single 'Full Medical Analysis'.\n\n"
            f"Expert Model A Analysis:\n{content_a}\n\n"
            f"Expert Model B Analysis:\n{content_b}\n\n"
            f"Original Patient Query: {context.get('prompt')}\n\n"
            "Produce a structured medical report with the following sections:\n"
            "1. CLINICAL SUMMARY (High-level state)\n"
            "2. EVIDENCE SYNTHESIS (Merging Model A and B's findings with PubMed context)\n"
            "3. FINAL CONSENSUS CONCLUSION (Primary diagnostic/management direction)\n"
            "4. CRITICAL SAFETY CAVEATS (GxP mandatory warnings)\n\n"
            "Ensure the tone is professional, exhaustive, and adheres to medical hedging standards."
        )
        
        synth_context = context.copy()
        synth_context["prompt"] = synthesis_prompt
        
        final_res = await self.arbiter.generate(synth_context)
        return final_res

    def _extract(self, result: dict | str) -> str:
        if isinstance(result, str):
            return result
        if "content" in result and isinstance(result["content"], str):
            return result["content"]
        if "choices" in result:
            return result["choices"][0]["message"]["content"]
        if "candidates" in result:
            return result["candidates"][0]["content"]["parts"][0]["text"]
        return str(result)
