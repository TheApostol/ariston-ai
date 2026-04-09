import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from vinci_core.models.anthropic_model import AnthropicModel
from vinci_core.models.openrouter_model import OpenRouterModel
from vinci_core.models.gemini_model import GeminiModel


@pytest.mark.asyncio
async def test_openrouter_format():
    model = OpenRouterModel()
    with patch("httpx.AsyncClient.post") as mock_post, \
         patch.dict("os.environ", {"OPENROUTER_API_KEY": "test_key"}):

        mock_response = MagicMock()
        mock_response.json = MagicMock(return_value={
            "choices": [{"message": {"content": "mock OR answer"}}]
        })
        mock_post.return_value = mock_response

        res = await model.generate(messages=[{"role": "user", "content": "test"}])
        assert "content" in res
        assert res["content"] == "mock OR answer"
        assert "model" in res
        assert "usage" in res
        assert "metadata" in res


@pytest.mark.asyncio
async def test_gemini_format():
    model = GeminiModel()
    mock_response = MagicMock()
    mock_response.text = "gemini answer"
    mock_response.usage_metadata = None

    with patch("asyncio.to_thread", new=AsyncMock(return_value=mock_response)):
        res = await model.generate(messages=[{"role": "user", "content": "test"}])
        assert res["content"] == "gemini answer"
        assert res["model"] == "gemini-1.5-flash"
        assert "usage" in res
        assert "metadata" in res
