"""
Consensus Router — two Claude models in parallel, OpenRouter arbiter synthesizes.

Claude Sonnet = deep reasoning.
Claude Haiku  = fast second opinion.

Observability: structured log lines per call, latency tracking.
Resilience: individual model failures fall back to Sonnet-only.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from config import settings

from vinci_core.models.openrouter_model import OpenRouterModel
from vinci_core.models.gemini_model import GeminiModel
from vinci_core.utils.retry import async_retry

logger = logging.getLogger("ariston.consensus")

_SONNET = "claude-sonnet-4-6"
_HAIKU = "claude-haiku-4-5-20251001"
_MAX_TOKENS = 2048
_TIMEOUT_SECONDS = 60


class ConsensusRouter:
    def __init__(self):
        self._client = None  # lazy-initialized
        self.arbiter = OpenRouterModel()
        self.gemini = GeminiModel()

    def _get_client(self):
        if self._client is not None:
            return self._client
        api_key = settings.ANTHROPIC_API_KEY
        if not api_key:
            return None
        try:
            import anthropic  # lazy import
            self._client = anthropic.AsyncAnthropic(
                api_key=api_key,
                timeout=_TIMEOUT_SECONDS,
            )
        except (ImportError, Exception) as e:
            logger.warning("[consensus] anthropic client init failed: %s", e)
        return self._client

    async def run(self, messages: List[Dict[str, Any]], prompt: str) -> Dict[str, Any]:
        """
        Run dual-Claude consensus + OpenRouter synthesis.

        Returns a normalised provider dict: { model, content, usage, metadata }.
        """
        start = time.monotonic()

        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        user_msgs = [m for m in messages if m["role"] != "system"]
        system = "\n\n".join(system_parts) or None

        client = self._get_client()

        # If no Anthropic client, fall straight through to Gemini consensus
        if not client:
            logger.warning("[consensus] no Anthropic client — delegating to Gemini")
            try:
                result = await self.gemini.generate(messages=messages)
                result.setdefault("metadata", {})["consensus"] = False
                result["metadata"]["provider"] = "google"
                return result
            except Exception as ge:
                return {
                    "model": "vinci-fallback",
                    "content": "Clinical consensus unavailable — all providers failed.",
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    "metadata": {"consensus": False, "error": True},
                }

        @async_retry(max_attempts=2, base_delay=1.0, exceptions=(Exception,))
        async def _call(model: str) -> str:
            kwargs: Dict[str, Any] = {"model": model, "max_tokens": _MAX_TOKENS, "messages": user_msgs}
            if system:
                kwargs["system"] = system
            r = await client.messages.create(**kwargs)
            return r.content[0].text

        # Run Sonnet and Haiku in parallel
        sonnet_text: Optional[str] = None
        haiku_text: Optional[str] = None
        consensus_achieved = False

        try:
            sonnet_text, haiku_text = await asyncio.gather(
                _call(_SONNET), _call(_HAIKU)
            )
            consensus_achieved = True
            logger.info(
                '{"event":"consensus_parallel_ok","latency_ms":%d}',
                round((time.monotonic() - start) * 1000),
            )
        except Exception as exc:
            logger.warning(
                '{"event":"consensus_parallel_failed","error":"%s","fallback":"sonnet_only"}',
                type(exc).__name__,
            )
            try:
                sonnet_text = await _call(_SONNET)
            except Exception as fallback_exc:
                logger.warning(
                    '{"event":"consensus_sonnet_fallback_failed","error":"%s","trying":"gemini"}',
                    type(fallback_exc).__name__,
                )
                try:
                    result = await self.gemini.generate(messages=messages)
                    result.setdefault("metadata", {})["consensus"] = False
                    return result
                except Exception as ge:
                    logger.error('{"event":"consensus_all_failed","gemini_error":"%s"}', ge)
                    return {
                        "model": _SONNET,
                        "content": (
                            "Clinical consensus service is temporarily unavailable. "
                            "Please retry your query."
                        ),
                        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                        "metadata": {"consensus": False, "error": True},
                    }

        # If only Sonnet succeeded, return it directly without synthesis
        if not consensus_achieved or not haiku_text:
            return {
                "model": _SONNET,
                "content": sonnet_text or "",
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "metadata": {
                    "consensus": False,
                    "models_used": [_SONNET],
                    "provider": "anthropic",
                },
            }

        # Synthesize via OpenRouter arbiter
        synthesis_prompt = (
            "You are the Ariston AI Clinical Consensus Arbiter.\n\n"
            f"Expert A (Sonnet):\n{sonnet_text}\n\n"
            f"Expert B (Haiku):\n{haiku_text}\n\n"
            f"Original Query:\n{prompt}\n\n"
            "Synthesize into a structured report:\n"
            "1. CLINICAL SUMMARY\n2. EVIDENCE SYNTHESIS\n"
            "3. FINAL CONSENSUS\n4. SAFETY CAVEATS\n\n"
            "Be professional and maintain medical hedging standards."
        )

        final = await self.arbiter.generate(
            messages=[{"role": "user", "content": synthesis_prompt}]
        )
        final_meta = final.get("metadata") or {}
        final_meta.update({
            "consensus": True,
            "models_used": [_SONNET, _HAIKU],
            "latency_ms": round((time.monotonic() - start) * 1000),
        })
        final["metadata"] = final_meta

        logger.info(
            '{"event":"consensus_synthesis_ok","latency_ms":%d}',
            final_meta["latency_ms"],
        )
        return final

