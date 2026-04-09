import pytest
from unittest.mock import MagicMock
from vinci_core.routing.model_router import ModelRouter
from vinci_core.models.anthropic_model import AnthropicModel
from vinci_core.models.openrouter_model import OpenRouterModel
from vinci_core.models.gemini_model import GeminiModel
from vinci_core.models.ollama_model import OllamaModel
from vinci_core.routing.consensus_router import ConsensusRouter, ConsensusModel


def test_consensus_model_alias():
    """ConsensusModel must be the same class as ConsensusRouter."""
    assert ConsensusModel is ConsensusRouter


def test_model_selection():
    router = ModelRouter()

    # Clinical → ConsensusRouter
    model = router.select_model("clinical", {})
    assert isinstance(model, ConsensusRouter)

    # Pharma → Anthropic
    model = router.select_model("pharma", {})
    assert isinstance(model, AnthropicModel)

    # General → OpenRouter
    model = router.select_model("general", {})
    assert isinstance(model, OpenRouterModel)


def test_fallback_logic():
    router = ModelRouter()

    # OpenRouter falls back to Gemini
    assert isinstance(router.get_fallback_model(router.openrouter), GeminiModel)

    # Gemini falls back to Anthropic
    assert isinstance(router.get_fallback_model(router.gemini), AnthropicModel)

    # Anthropic falls back to OpenRouter
    assert isinstance(router.get_fallback_model(router.anthropic), OpenRouterModel)

    # OllamaModel instance → last resort (returns ollama itself since it's not in primary map)
    fallback = router.get_fallback_model(router.ollama)
    assert isinstance(fallback, OllamaModel)
