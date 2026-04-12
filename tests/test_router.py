"""
Model router tests.

The ModelRouter delegates to four strategies:
  - "consensus"  → ConsensusRouter  (clinical layer)
  - "anthropic"  → AnthropicModel   (pharma / latam)
  - "openai"     → OpenAIModel      (data layer)
  - "gemini"     → GeminiModel      (radiology / base / general)

Tests verify layer→strategy mapping via the layer_model_map dict.
"""

import pytest
from vinci_core.routing.model_router import ModelRouter
from vinci_core.models.anthropic_model import AnthropicModel
from vinci_core.models.openrouter_model import OpenRouterModel
from vinci_core.models.openai_model import OpenAIModel
from vinci_core.routing.consensus_router import ConsensusRouter


def test_clinical_layer_uses_consensus():
    router = ModelRouter()
    strategy = router.layer_model_map.get("clinical")
    assert strategy == "consensus"


def test_pharma_layer_uses_anthropic():
    router = ModelRouter()
    strategy = router.layer_model_map.get("pharma")
    assert strategy == "anthropic"


def test_data_layer_uses_openai():
    router = ModelRouter()
    strategy = router.layer_model_map.get("data")
    assert strategy == "openai"


def test_radiology_layer_uses_gemini():
    router = ModelRouter()
    strategy = router.layer_model_map.get("radiology")
    assert strategy == "gemini"


def test_general_layer_uses_gemini():
    router = ModelRouter()
    strategy = router.layer_model_map.get("general")
    assert strategy == "gemini"


def test_base_layer_uses_gemini():
    router = ModelRouter()
    strategy = router.layer_model_map.get("base")
    assert strategy == "gemini"


def test_all_layers_have_model_map():
    router = ModelRouter()
    for layer in router.layers:
        assert layer in router.layer_model_map, f"Layer '{layer}' missing from layer_model_map"


def test_router_has_required_provider_instances():
    router = ModelRouter()
    assert isinstance(router.anthropic, AnthropicModel)
    assert isinstance(router.openrouter, OpenRouterModel)
    assert isinstance(router.openai, OpenAIModel)
    assert isinstance(router.consensus, ConsensusRouter)


def test_consensus_router_has_correct_models():
    from vinci_core.routing.consensus_router import _SONNET, _HAIKU
    assert "claude-sonnet" in _SONNET
    assert "claude-haiku" in _HAIKU

