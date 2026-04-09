"""
Model Router — layer-based model selection.
clinical  → ConsensusRouter (Sonnet + Haiku + arbiter)
pharma    → Anthropic Sonnet
data      → Anthropic Sonnet
radiology → Anthropic Sonnet (until Harrison.ai partnership)
general   → OpenRouter (free)
Fallback chain: primary → OpenRouter → Gemini → Ollama → error
"""

import logging
from vinci_core.models.anthropic_model import AnthropicModel
from vinci_core.models.openrouter_model import OpenRouterModel
from vinci_core.models.gemini_model import GeminiModel
from vinci_core.models.ollama_model import OllamaModel
from vinci_core.routing.consensus_router import ConsensusRouter
from vinci_core.layers.base_layer import BaseLayer
from vinci_core.layers.pharma_layer import PharmaLayer
from vinci_core.layers.clinical_layer import ClinicalLayer
from vinci_core.layers.data_layer import DataLayer

logger = logging.getLogger("ariston.router")


class ModelRouter:
    def __init__(self):
        self.anthropic = AnthropicModel()
        self.openrouter = OpenRouterModel()
        self.gemini = GeminiModel()
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

        # Ordered fallback chain per primary model type
        self._fallback_chain = {
            "consensus":  [self.anthropic, self.openrouter, self.gemini, self.ollama],
            "anthropic":  [self.openrouter, self.gemini, self.ollama],
            "openrouter": [self.gemini, self.ollama],
            "gemini":     [self.anthropic, self.ollama],
        }

    # ------------------------------------------------------------------
    # Public helpers (used by tests and external callers)
    # ------------------------------------------------------------------

    def select_model(self, layer: str, context: dict) -> object:
        """Return the primary model/router instance for a given layer."""
        model_key = self.layer_model_map.get(layer, "openrouter")
        if model_key == "consensus":
            return self.consensus
        if model_key == "anthropic":
            return self.anthropic
        if model_key == "gemini":
            return self.gemini
        return self.openrouter

    def get_fallback_model(self, model_instance) -> object:
        """Return the first fallback for a given model instance."""
        # Determine the key for the given instance
        for key, primary in [
            ("consensus", self.consensus),
            ("anthropic", self.anthropic),
            ("openrouter", self.openrouter),
            ("gemini", self.gemini),
        ]:
            if model_instance is primary or isinstance(model_instance, type(primary)):
                chain = self._fallback_chain.get(key, [])
                return chain[0] if chain else self.ollama
        return self.ollama

    # ------------------------------------------------------------------
    # Core execution
    # ------------------------------------------------------------------

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
        effective_model = model or self.layer_model_map.get(layer, "openrouter")

        try:
            return await self._call(effective_model, messages=messages, prompt=prompt)
        except Exception as exc:
            logger.warning(
                "[router] request_id=%s primary=%s failed (%s); trying fallback chain",
                request_id, effective_model, exc,
            )
            for fallback in self._fallback_chain.get(effective_model, [self.openrouter]):
                try:
                    result = await fallback.generate(messages=messages)
                    result.setdefault("metadata", {})["fallback_used"] = True
                    return result
                except Exception as fb_exc:
                    logger.warning(
                        "[router] request_id=%s fallback=%s failed: %s",
                        request_id, type(fallback).__name__, fb_exc,
                    )
            return {
                "model": "vinci",
                "content": "All providers unavailable. Please try again later.",
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "metadata": {"error": True, "fallback_used": True},
            }

    async def _call(self, effective_model: str, messages: list, prompt: str) -> dict:
        if effective_model == "consensus":
            return await self.consensus.run(messages=messages, prompt=prompt)
        if effective_model == "anthropic":
            return await self.anthropic.generate(messages=messages)
        if effective_model == "gemini":
            return await self.gemini.generate(messages=messages)
        return await self.openrouter.generate(messages=messages)
