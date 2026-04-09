"""
Model Router — layer-based model selection.
clinical  → ConsensusRouter (Sonnet + Haiku + arbiter)
pharma    → Anthropic Sonnet
data      → Anthropic Sonnet
radiology → Anthropic Sonnet (until Harrison.ai partnership)
general   → OpenRouter (free)
Fallback: primary → OpenRouter → error response
"""

from vinci_core.models.anthropic_model import AnthropicModel
from vinci_core.models.openrouter_model import OpenRouterModel
from vinci_core.routing.consensus_router import ConsensusRouter
from vinci_core.layers.base_layer import BaseLayer
from vinci_core.layers.pharma_layer import PharmaLayer
from vinci_core.layers.clinical_layer import ClinicalLayer
from vinci_core.layers.data_layer import DataLayer


class ModelRouter:
    def __init__(self):
        self.anthropic = AnthropicModel()
        self.openrouter = OpenRouterModel()
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

    async def run(self, prompt: str, model: str = None, layer: str = "base", context: dict = None) -> dict:
        layer_obj = self.layers.get(layer, self.layers["base"])
        messages = layer_obj.build_messages(prompt, context)
        effective_model = model or self.layer_model_map.get(layer, "openrouter")

        try:
            if effective_model == "consensus":
                return await self.consensus.run(messages=messages, prompt=prompt)
            elif effective_model == "anthropic":
                return await self.anthropic.generate(messages=messages)
            else:
                return await self.openrouter.generate(messages=messages)
        except Exception as e:
            print(f"[Router] {effective_model} failed, fallback → OpenRouter: {e}")
            return await self.openrouter.generate(messages=messages)
