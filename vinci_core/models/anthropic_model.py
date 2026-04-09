"""
Anthropic Claude provider — normalized interface with timeout and retry.

Returns: { model, content, usage, metadata }
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

import anthropic
from config import settings

from vinci_core.models.base_model import BaseModel
from vinci_core.utils.retry import async_retry

logger = logging.getLogger("ariston.providers.anthropic")

_DEFAULT_MODEL = "claude-sonnet-4-6"
_MAX_TOKENS = 2048
_TIMEOUT_SECONDS = 60


class AnthropicModel(BaseModel):
    name = "anthropic"

    def __init__(self):
        self.client = anthropic.AsyncAnthropic(
            api_key=settings.ANTHROPIC_API_KEY,
            timeout=_TIMEOUT_SECONDS,
        )

    @async_retry(max_attempts=3, base_delay=1.0, exceptions=(anthropic.APIStatusError, anthropic.APIConnectionError))
    async def generate(
        self,
        messages: Optional[List[Dict[str, str]]] = None,
        prompt: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Generate a completion via Anthropic Claude.

        Accepts either *messages* (OpenAI-style list) or a plain *prompt* string.
        System messages are extracted and passed via Anthropic's dedicated param.
        """
        if not messages and prompt:
            messages = [{"role": "user", "content": prompt}]
        elif not messages:
            messages = []

        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        user_messages = [m for m in messages if m["role"] != "system"]
        system_prompt = "\n\n".join(system_parts) if system_parts else None

        target_model = model or _DEFAULT_MODEL
        kwargs_call: Dict[str, Any] = {
            "model": target_model,
            "max_tokens": _MAX_TOKENS,
            "messages": user_messages,
        }
        if system_prompt:
            kwargs_call["system"] = system_prompt

        logger.debug("[anthropic] calling model=%s messages=%d", target_model, len(user_messages))
        response = await self.client.messages.create(**kwargs_call)

        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

        return {
            "model": target_model,
            "content": response.content[0].text,
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
            "metadata": {"provider": "anthropic"},
        }

