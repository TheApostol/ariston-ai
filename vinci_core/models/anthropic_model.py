import asyncio
import logging
from config import settings

logger = logging.getLogger("ariston.anthropic")

_TIMEOUT_SECONDS = 60
_MAX_RETRIES = 3


class AnthropicModel:
    name = "anthropic"
    DEFAULT_MODEL = "claude-sonnet-4-6"

    def __init__(self):
        self._client = None  # lazy-initialized

    def _get_client(self):
        if self._client is not None:
            return self._client
        api_key = settings.ANTHROPIC_API_KEY
        if not api_key:
            return None
        try:
            import anthropic  # lazy import — SDK optional
            self._client = anthropic.AsyncAnthropic(
                api_key=api_key,
                timeout=_TIMEOUT_SECONDS,
                max_retries=_MAX_RETRIES,
            )
        except (ImportError, Exception) as e:
            logger.warning("[anthropic] client init failed: %s", e)
        return self._client

    async def generate(self, messages: list, model: str = None) -> dict:
        client = self._get_client()
        if not client:
            raise RuntimeError("Anthropic client unavailable — check ANTHROPIC_API_KEY and anthropic SDK")

        used_model = model or self.DEFAULT_MODEL

        # Strip system messages — Anthropic uses a separate system param
        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        user_messages = [m for m in messages if m["role"] != "system"]
        system_prompt = "\n\n".join(system_parts) if system_parts else None

        kwargs = {
            "model": used_model,
            "max_tokens": 2048,
            "messages": user_messages,
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        try:
            response = await asyncio.wait_for(
                client.messages.create(**kwargs),
                timeout=_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.error("[Anthropic] timeout after %ds for model=%s", _TIMEOUT_SECONDS, used_model)
            raise TimeoutError(f"Anthropic timed out after {_TIMEOUT_SECONDS}s")

        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

        return {
            "model": used_model,
            "content": response.content[0].text,
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
            "metadata": {
                "provider": "anthropic",
                "stop_reason": response.stop_reason,
            },
        }
