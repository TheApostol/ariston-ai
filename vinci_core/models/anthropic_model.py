import asyncio
import logging
import anthropic
from config import settings

logger = logging.getLogger("ariston.anthropic")

_TIMEOUT_SECONDS = 60
_MAX_RETRIES = 3
_RETRY_STATUS_CODES = {429, 502, 503, 529}


class AnthropicModel:
    name = "anthropic"
    DEFAULT_MODEL = "claude-sonnet-4-6"

    def __init__(self):
        self.client = anthropic.AsyncAnthropic(
            api_key=settings.ANTHROPIC_API_KEY,
            timeout=_TIMEOUT_SECONDS,
            max_retries=_MAX_RETRIES,
        )

    async def generate(self, messages: list, model: str = None) -> dict:
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
                self.client.messages.create(**kwargs),
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
