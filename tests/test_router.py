import pytest
from vinci_core.routing.model_router import ModelRouter
from vinci_core.models.anthropic_model import AnthropicModel
from vinci_core.models.openrouter_model import OpenRouterModel
from vinci_core.models.gemini_model import GeminiModel
from vinci_core.routing.consensus_router import ConsensusModel

def test_model_selection():
    router = ModelRouter()
    
    # Clinical -> Consensus
    model = router.select_model("clinical", {})
    assert isinstance(model, ConsensusModel)
    
    # Pharma -> Anthropic
    model = router.select_model("pharma", {})
    assert isinstance(model, AnthropicModel)
    
    # Normal -> OpenRouter
    model = router.select_model("general", {})
    assert isinstance(model, OpenRouterModel)

def test_fallback_logic():
    router = ModelRouter()
    
    # OpenRouter falls back to Gemini
    assert isinstance(router.get_fallback_model(OpenRouterModel()), GeminiModel)
    
    # Gemini falls back to Anthropic
    assert isinstance(router.get_fallback_model(GeminiModel()), AnthropicModel)
    
    # Anthropic now falls back to local Ollama
    from vinci_core.models.ollama_model import OllamaModel
    assert isinstance(router.get_fallback_model(AnthropicModel()), OllamaModel)
