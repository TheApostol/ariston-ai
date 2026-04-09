from vinci_core.models.openrouter_model import OpenRouterModel
from vinci_core.models.anthropic_model import AnthropicModel
from vinci_core.models.gemini_model import GeminiModel
from vinci_core.routing.consensus_router import ConsensusModel
from vinci_core.models.ollama_model import OllamaModel


class ModelRouter:
    def __init__(self):
        self.openrouter = OpenRouterModel()
        self.anthropic = AnthropicModel()
        self.gemini = GeminiModel()
        self.consensus = ConsensusModel()
        self.local_ollama = None # Lazy load due to size and dependency

    def select_model(self, layer: str, context: dict):
        # Privacy-first local execution overrides all network models
        if layer == "local":
            if not self.local_ollama:
                self.local_ollama = OllamaModel()
            return self.local_ollama

        # PRIORITY: free / available models first

        if layer == "clinical":
            return self.consensus

        if layer == "pharma":
            return self.anthropic

        if layer == "data":
            return self.gemini

        # DEFAULT → OpenRouter (cheapest/free option)
        return self.openrouter

    def get_fallback_model(self, failed_model):
        
        # Privacy-first local execution fallback
        # If all cloud providers fail, we attempt to use local Ollama
        
        if isinstance(failed_model, OpenRouterModel):
            return self.gemini
            
        if isinstance(failed_model, GeminiModel):
            return self.anthropic
            
        if isinstance(failed_model, AnthropicModel):
            # Final fallback to local inference
            if not self.local_ollama:
                try:
                    self.local_ollama = OllamaModel()
                    return self.local_ollama
                except:
                    return None
            return self.local_ollama

        return None
