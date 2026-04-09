"""
OpenRouter provider — normalized interface with timeout and retry.

Returns: { model, content, usage, metadata }
"""

import logging
import os
from typing import Any, Dict, List, Optional

import httpx

from vinci_core.models.base_model import BaseModel
from vinci_core.utils.retry import async_retry

logger = logging.getLogger("ariston.providers.openrouter")

_DEFAULT_MODEL = "openrouter/auto"
_TIMEOUT_SECONDS = 30
_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterModel(BaseModel):
    name = "openrouter"

    def __init__(self):
        self._api_key: Optional[str] = None

    def _get_api_key(self) -> str:
        if not self._api_key:
            key = os.getenv("OPENROUTER_API_KEY")
            if not key:
                raise ValueError("Missing OPENROUTER_API_KEY environment variable")
            self._api_key = key
        return self._api_key

    @async_retry(max_attempts=3, base_delay=1.0, exceptions=(httpx.TimeoutException, httpx.NetworkError))
    async def generate(
        self,
        messages: Optional[List[Dict[str, str]]] = None,
        prompt: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Generate a completion via OpenRouter.

        Accepts either *messages* list or a plain *prompt* string.
        """
        if not messages and prompt:
            messages = [{"role": "user", "content": prompt}]
        elif not messages:
            messages = []

        target_model = model or _DEFAULT_MODEL

        headers = {
            "Authorization": f"Bearer {self._get_api_key()}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://ariston.ai",
            "X-Title": "Ariston AI",
        }
        payload = {"model": target_model, "messages": messages}

        logger.debug("[openrouter] calling model=%s messages=%d", target_model, len(messages))

        async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
            response = await client.post(_BASE_URL, headers=headers, json=payload)

        try:
            data = response.json()
        except Exception:
            return _error_response(f"Invalid JSON response (HTTP {response.status_code})")

        if "error" in data:
            err = data["error"]
            msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            logger.warning("[openrouter] API error: %s", msg)
            return _error_response(f"OpenRouter error: {msg}")

        raw_usage = data.get("usage") or {}
        prompt_tokens = raw_usage.get("prompt_tokens", 0) or 0
        completion_tokens = raw_usage.get("completion_tokens", 0) or 0

        return {
            "model": data.get("model", target_model),
            "content": data["choices"][0]["message"]["content"],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "metadata": {"provider": "openrouter"},
        }


def _error_response(message: str) -> Dict[str, Any]:
    return {
        "model": "openrouter",
        "content": message,
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "metadata": {"provider": "openrouter", "error": True},
    }

