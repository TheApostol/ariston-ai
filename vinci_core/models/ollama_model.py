"""
Ollama local model provider — normalized interface with timeout and retry.

Returns: { model, content, usage, metadata }

Ollama must be running locally on port 11434.
Falls back gracefully if the server is unavailable.
"""

import logging
from typing import Any, Dict, List, Optional

import httpx

from vinci_core.models.base_model import BaseModel
from vinci_core.utils.retry import async_retry

logger = logging.getLogger("ariston.providers.ollama")

_DEFAULT_MODEL = "llama3:8b"
_BASE_URL = "http://localhost:11434/api/generate"
_TIMEOUT_SECONDS = 60


class OllamaModel(BaseModel):
    name = "ollama"

    @async_retry(max_attempts=2, base_delay=0.5, exceptions=(httpx.TimeoutException, httpx.ConnectError))
    async def generate(
        self,
        messages: Optional[List[Dict[str, str]]] = None,
        prompt: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Generate a completion via a local Ollama server.

        Accepts either *messages* list (flattened to string) or a plain *prompt*.
        """
        if messages:
            content = "\n\n".join(
                f"[{m['role'].upper()}] {m['content']}" for m in messages
            )
        elif prompt:
            content = prompt
        else:
            content = ""

        target_model = model or _DEFAULT_MODEL

        logger.debug("[ollama] calling model=%s prompt_len=%d", target_model, len(content))

        async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
            response = await client.post(
                _BASE_URL,
                json={
                    "model": target_model,
                    "prompt": content,
                    "stream": False,
                    "options": {"temperature": 0.7, "num_predict": 512},
                    "keep_alive": "10m",
                },
            )

        data = response.json()
        generated = data.get("response", "")

        # Ollama returns eval_count (output tokens) and prompt_eval_count
        prompt_tokens = data.get("prompt_eval_count", 0) or 0
        output_tokens = data.get("eval_count", 0) or 0

        return {
            "model": target_model,
            "content": generated,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": prompt_tokens + output_tokens,
            },
            "metadata": {"provider": "ollama"},
        }
