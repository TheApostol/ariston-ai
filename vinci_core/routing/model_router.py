"""
Model Router — layer-based model selection with scoring and LATAM support.

Layer → Model mapping:
  clinical  → ConsensusRouter (Sonnet + Haiku + arbiter)
  pharma    → Anthropic Sonnet
  latam     → Anthropic Sonnet (regulatory precision required)
  data      → OpenAI (analysis tasks)
  radiology → Gemini (vision + clinical multimodal)
  general   → OpenRouter (free)

Fallback chain: primary → OpenRouter → Gemini → OpenAI → structured error response
"""

import logging
from vinci_core.models.anthropic_model import AnthropicModel
from vinci_core.models.openrouter_model import OpenRouterModel
from vinci_core.models.gemini_model import GeminiModel
from vinci_core.models.openai_model import OpenAIModel
from vinci_core.routing.consensus_router import ConsensusRouter
from vinci_core.layers.base_layer import BaseLayer
from vinci_core.layers.pharma_layer import PharmaLayer
from vinci_core.layers.clinical_layer import ClinicalLayer
from vinci_core.layers.data_layer import DataLayer
from vinci_core.layers.latam_layer import LatamLayer

logger = logging.getLogger("ariston.router")


class ModelRouter:
    def __init__(self):
        self.anthropic = AnthropicModel()
        self.openrouter = OpenRouterModel()
        self.gemini = GeminiModel()
        self.openai = OpenAIModel()
        self.consensus = ConsensusRouter()

        self.layers = {
            "base":      BaseLayer(),
            "pharma":    PharmaLayer(),
            "clinical":  ClinicalLayer(),
            "data":      DataLayer(),
            "radiology": ClinicalLayer(),
            "latam":     LatamLayer(),
            "general":   BaseLayer(),
        }

        # Scoring: higher = more capable/expensive provider used
        self.layer_model_map = {
            "clinical":  "consensus",   # score: 10 — dual-model + arbiter
            "latam":     "anthropic",   # score: 8 — regulatory precision
            "pharma":    "anthropic",   # score: 8 — structured science
            "data":      "openai",      # score: 7 — analysis tasks (GPT-4o-mini)
            "radiology": "gemini",      # score: 7 — vision + clinical (Gemini multimodal)
            "base":      "gemini",      # score: 5 — general purpose
            "general":   "gemini",      # score: 5 — general purpose
        }

    def _select_model(self, layer: str, override: str = None) -> str:
        """Select model based on layer scoring. Override takes precedence."""
        if override:
            return override
        return self.layer_model_map.get(layer, "openrouter")

    async def run(
        self,
        prompt: str,
        model: str = None,
        layer: str = "base",
        context: dict = None,
        request_id: str = None,
    ) -> dict:
        layer_obj = self.layers.get(layer, self.layers["base"])
        messages = layer_obj.build_messages(prompt, context)
        effective_model = self._select_model(layer, override=model)
        fallback_used = False

        try:
            if effective_model == "consensus":
                result = await self.consensus.run(messages=messages, prompt=prompt)
            elif effective_model == "anthropic":
                result = await self.anthropic.generate(messages=messages)
            elif effective_model == "gemini":
                result = await self.gemini.generate(messages=messages)
            elif effective_model == "openai":
                result = await self.openai.generate(messages=messages)
            else:
                result = await self.openrouter.generate(messages=messages)

        except Exception as e:
            fallback_used = True
            logger.warning(
                "[Router] request_id=%s provider=%s failed, fallback→OpenRouter: %s",
                request_id, effective_model, e,
            )
            try:
                result = await self.openrouter.generate(messages=messages)
                result.setdefault("metadata", {})["fallback_used"] = True
                result["metadata"]["fallback_reason"] = str(e)
            except Exception as fe:
                logger.warning("[Router] OpenRouter also failed, fallback→Gemini: %s", fe)
                try:
                    result = await self.gemini.generate(messages=messages)
                    result.setdefault("metadata", {})["fallback_used"] = True
                    result["metadata"]["fallback_reason"] = f"Primary: {e} | OpenRouter: {fe}"
                except Exception as ge:
                    logger.warning("[Router] Gemini also failed, fallback→OpenAI: %s", ge)
                    try:
                        result = await self.openai.generate(messages=messages)
                        result.setdefault("metadata", {})["fallback_used"] = True
                        result["metadata"]["fallback_reason"] = (
                            f"Primary: {e} | OpenRouter: {fe} | Gemini: {ge}"
                        )
                    except Exception as oe:
                        logger.error("[Router] All providers failed. OpenAI: %s", oe)
                        return {
                            "model": "vinci-fallback",
                            "content": "All providers unavailable. Please retry.",
                            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                            "metadata": {
                                "error": True,
                                "fallback_used": True,
                                "fallback_reason": (
                                    f"Primary: {e} | OpenRouter: {fe} | Gemini: {ge} | OpenAI: {oe}"
                                ),
                            },
                        }

        result.setdefault("metadata", {})
        result["metadata"]["fallback_used"] = fallback_used
        result["metadata"]["layer"] = layer
        if request_id:
            result["metadata"]["request_id"] = request_id

        return result
