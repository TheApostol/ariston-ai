"""
Model Router — layer-based model selection with observability and fallback chain.

Layer → model strategy:
  clinical  → ConsensusRouter (Sonnet + Haiku + arbiter)
  pharma    → Anthropic Sonnet
  data      → Anthropic Sonnet
  radiology → Anthropic Sonnet (until Harrison.ai partnership)
  general   → OpenRouter (free tier)
  base      → OpenRouter (free tier)

Fallback chain: primary → OpenRouter → Ollama (local) → error response
Every call emits structured log lines and returns `fallback_used` in metadata.
"""

import logging
import time
from typing import Any, Dict, Optional

from vinci_core.models.anthropic_model import AnthropicModel
from vinci_core.models.openrouter_model import OpenRouterModel
from vinci_core.models.ollama_model import OllamaModel
from vinci_core.routing.consensus_router import ConsensusRouter
from vinci_core.layers.base_layer import BaseLayer
from vinci_core.layers.pharma_layer import PharmaLayer
from vinci_core.layers.clinical_layer import ClinicalLayer
from vinci_core.layers.data_layer import DataLayer

logger = logging.getLogger("ariston.model_router")

_FALLBACK_ERROR_CONTENT = (
    "All model providers are currently unavailable. "
    "Please try again shortly or contact support."
)


class ModelRouter:
    def __init__(self):
        self.anthropic = AnthropicModel()
        self.openrouter = OpenRouterModel()
        self.ollama = OllamaModel()
        self.consensus = ConsensusRouter()

        self.layers = {
            "base":      BaseLayer(),
            "pharma":    PharmaLayer(),
            "clinical":  ClinicalLayer(),
            "data":      DataLayer(),
            "radiology": ClinicalLayer(),
            "general":   BaseLayer(),
        }

        self.layer_model_map = {
            "clinical":  "consensus",
            "pharma":    "anthropic",
            "data":      "anthropic",
            "radiology": "anthropic",
            "base":      "openrouter",
            "general":   "openrouter",
        }

    async def run(
        self,
        prompt: str,
        model: Optional[str] = None,
        layer: str = "base",
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        layer_obj = self.layers.get(layer, self.layers["base"])
        messages = layer_obj.build_messages(prompt, context)
        effective_model = model or self.layer_model_map.get(layer, "openrouter")

        start = time.monotonic()

        logger.info(
            '{"event":"router_dispatch","layer":"%s","strategy":"%s"}',
            layer, effective_model,
        )

        # ── Primary call ───────────────────────────────────────────────────────
        try:
            result = await self._call_strategy(effective_model, messages, prompt)
            latency_ms = round((time.monotonic() - start) * 1000)
            _tag_result(result, fallback_used=False, latency_ms=latency_ms)
            logger.info(
                '{"event":"router_ok","strategy":"%s","latency_ms":%d}',
                effective_model, latency_ms,
            )
            return result
        except Exception as primary_exc:
            logger.warning(
                '{"event":"router_primary_failed","strategy":"%s","error":"%s","fallback":"openrouter"}',
                effective_model, type(primary_exc).__name__,
            )

        # ── Fallback 1: OpenRouter ─────────────────────────────────────────────
        if effective_model != "openrouter":
            try:
                result = await self.openrouter.generate(messages=messages)
                latency_ms = round((time.monotonic() - start) * 1000)
                _tag_result(result, fallback_used=True, latency_ms=latency_ms)
                logger.info(
                    '{"event":"router_fallback_ok","strategy":"openrouter","latency_ms":%d}',
                    latency_ms,
                )
                return result
            except Exception as fb1_exc:
                logger.warning(
                    '{"event":"router_fallback1_failed","error":"%s","fallback":"ollama"}',
                    type(fb1_exc).__name__,
                )

        # ── Fallback 2: Ollama (local) ─────────────────────────────────────────
        try:
            result = await self.ollama.generate(messages=messages)
            latency_ms = round((time.monotonic() - start) * 1000)
            _tag_result(result, fallback_used=True, latency_ms=latency_ms)
            logger.info(
                '{"event":"router_fallback_ok","strategy":"ollama","latency_ms":%d}',
                latency_ms,
            )
            return result
        except Exception as fb2_exc:
            logger.error(
                '{"event":"router_all_failed","error":"%s"}',
                type(fb2_exc).__name__,
            )

        # ── Total failure — safe structured response ───────────────────────────
        latency_ms = round((time.monotonic() - start) * 1000)
        return {
            "model": "none",
            "content": _FALLBACK_ERROR_CONTENT,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "metadata": {
                "provider": "none",
                "error": True,
                "fallback_used": True,
                "latency_ms": latency_ms,
            },
        }

    async def _call_strategy(
        self, strategy: str, messages: list, prompt: str
    ) -> Dict[str, Any]:
        if strategy == "consensus":
            return await self.consensus.run(messages=messages, prompt=prompt)
        if strategy == "anthropic":
            return await self.anthropic.generate(messages=messages)
        return await self.openrouter.generate(messages=messages)


def _tag_result(result: Dict[str, Any], fallback_used: bool, latency_ms: int) -> None:
    """Mutate result in-place to stamp observability fields."""
    meta = result.setdefault("metadata", {})
    meta["fallback_used"] = fallback_used
    meta["latency_ms"] = latency_ms

