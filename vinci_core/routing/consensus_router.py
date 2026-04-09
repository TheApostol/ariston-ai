"""
Consensus Router — two Claude models in parallel, OpenRouter arbiter synthesizes.
Claude Sonnet = deep reasoning. Claude Haiku = fast second opinion. Cost-efficient.
"""

import asyncio
import anthropic
from config import settings
from vinci_core.models.openrouter_model import OpenRouterModel


class ConsensusRouter:
    def __init__(self):
        self.client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
        self.arbiter = OpenRouterModel()

    async def run(self, messages: list, prompt: str) -> dict:
        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        user_msgs = [m for m in messages if m["role"] != "system"]
        system = "\n\n".join(system_parts) or None

        async def call(model: str) -> str:
            kwargs = {"model": model, "max_tokens": 2048, "messages": user_msgs}
            if system:
                kwargs["system"] = system
            r = await self.client.messages.create(**kwargs)
            return r.content[0].text

        try:
            sonnet, haiku = await asyncio.gather(
                call("claude-sonnet-4-6"),
                call("claude-haiku-4-5-20251001"),
            )
        except Exception as e:
            print(f"[Consensus] parallel failed, single Sonnet fallback: {e}")
            return {"model": "claude-sonnet-4-6", "content": await call("claude-sonnet-4-6"),
                    "usage": {}, "metadata": {"consensus": False}}

        synthesis = (
            "You are the Ariston AI Clinical Consensus Arbiter.\n\n"
            f"Expert A (Sonnet):\n{sonnet}\n\n"
            f"Expert B (Haiku):\n{haiku}\n\n"
            f"Original Query:\n{prompt}\n\n"
            "Synthesize into a structured report:\n"
            "1. CLINICAL SUMMARY\n2. EVIDENCE SYNTHESIS\n"
            "3. FINAL CONSENSUS\n4. SAFETY CAVEATS\n\n"
            "Be professional and maintain medical hedging standards."
        )

        final = await self.arbiter.generate(messages=[{"role": "user", "content": synthesis}])
        final["metadata"] = final.get("metadata", {})
        final["metadata"].update({"consensus": True, "models_used": ["claude-sonnet-4-6", "claude-haiku-4-5-20251001"]})
        return final
