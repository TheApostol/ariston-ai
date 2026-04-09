"""
Google Gemini provider — normalized interface with async execution, timeout, and retry.

Returns: { model, content, usage, metadata }

Note: google-genai's `generate_content` is synchronous, so it is executed
in a thread pool via asyncio.to_thread to keep the event loop non-blocking.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from google import genai
from config import settings

from vinci_core.models.base_model import BaseModel
from vinci_core.utils.retry import async_retry

logger = logging.getLogger("ariston.providers.gemini")

_DEFAULT_MODEL = "gemini-1.5-flash"
_TIMEOUT_SECONDS = 45


class GeminiModel(BaseModel):
    name = "gemini"

    def __init__(self):
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)

    @async_retry(max_attempts=3, base_delay=1.5, exceptions=(Exception,))
    async def generate(
        self,
        messages: Optional[List[Dict[str, str]]] = None,
        prompt: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Generate a completion via Gemini.

        Converts messages list to a single prompt string (Gemini uses content string).
        Runs the synchronous SDK call in a thread pool to avoid blocking the event loop.
        """
        if messages:
            # Flatten messages into a single content string for Gemini
            content = "\n\n".join(
                f"[{m['role'].upper()}] {m['content']}" for m in messages
            )
        elif prompt:
            content = prompt
        else:
            content = ""

        target_model = model or _DEFAULT_MODEL
        client = self.client

        logger.debug("[gemini] calling model=%s content_len=%d", target_model, len(content))

        def _call() -> Any:
            return client.models.generate_content(model=target_model, contents=content)

        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(_call),
                timeout=_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError as exc:
            raise TimeoutError(f"Gemini request timed out after {_TIMEOUT_SECONDS}s") from exc

        # Normalize usage — Gemini may not always expose token counts
        usage_meta = getattr(response, "usage_metadata", None)
        prompt_tokens = getattr(usage_meta, "prompt_token_count", 0) or 0
        output_tokens = getattr(usage_meta, "candidates_token_count", 0) or 0

        return {
            "model": target_model,
            "content": response.text,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": prompt_tokens + output_tokens,
            },
            "metadata": {"provider": "google"},
        }

