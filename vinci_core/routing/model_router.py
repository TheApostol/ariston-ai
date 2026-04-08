from vinci_core.models.openrouter_model import OpenRouterModel
from vinci_core.models.anthropic_model import AnthropicModel
from vinci_core.models.gemini_model import GeminiModel


class ModelRouter:
    def __init__(self):
        self.openrouter = OpenRouterModel()
        self.anthropic = AnthropicModel()
        self.gemini = GeminiModel()

    def select_model(self, layer: str, context: dict):

        # PRIORITY: free / available models first

        if layer == "clinical":
            return self.anthropic

        if layer == "pharma":
            return self.anthropic

        if layer == "data":
            return self.gemini

        # DEFAULT → OpenRouter (cheapest/free option)
        return self.openrouter

    def get_fallback_model(self, failed_model):

        if isinstance(failed_model, OpenRouterModel):
            return self.gemini

        if isinstance(failed_model, GeminiModel):
            return self.anthropic

        if isinstance(failed_model, AnthropicModel):
            return None

        return None
