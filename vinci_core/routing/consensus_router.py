"""
Consensus Router — two Claude models in parallel, OpenRouter arbiter synthesizes.
Claude Sonnet = deep reasoning. Claude Haiku = fast second opinion. Cost-efficient.

`ConsensusModel` is provided as an alias for `ConsensusRouter` so external code
(e.g. tests) can import either name.
"""

import asyncio
import logging
import anthropic
from config import settings
from vinci_core.models.openrouter_model import OpenRouterModel

logger = logging.getLogger("ariston.consensus")


class ConsensusRouter:
    def __init__(self):
        self.client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
        self.arbiter = OpenRouterModel()

    async def run(self, messages: list, prompt: str, request_id: str = None) -> dict:
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
        except Exception as exc:
            logger.warning(
                "[consensus] request_id=%s parallel failed (%s); single Sonnet fallback",
                request_id, exc,
            )
            return {
                "model": "claude-sonnet-4-6",
                "content": await call("claude-sonnet-4-6"),
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "metadata": {"consensus": False, "fallback_used": True},
            }

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
        final.setdefault("metadata", {}).update({
            "consensus": True,
            "models_used": ["claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
        })
        return final


# Alias — preserves backward compatibility for any code importing ConsensusModel
ConsensusModel = ConsensusRouter
