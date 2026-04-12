"""
Google Gemini provider — normalized interface with async execution, timeout, and retry.

Returns: { model, content, usage, metadata }

Note: google-genai's `generate_content` is synchronous, so it is executed
in a thread pool via asyncio.to_thread to keep the event loop non-blocking.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from config import settings

from vinci_core.models.base_model import BaseModel
from vinci_core.utils.retry import async_retry

logger = logging.getLogger("ariston.providers.gemini")

_DEFAULT_MODEL = "gemini-2.0-flash"
_TIMEOUT_SECONDS = 45


class GeminiModel(BaseModel):
    name = "gemini"

    def __init__(self):
        self._client = None  # lazy-initialized

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from google import genai  # lazy import — SDK optional
            self._client = genai.Client(api_key=settings.GEMINI_API_KEY)
        except (ImportError, Exception) as e:
            logger.warning("[gemini] client init failed: %s", e)
        return self._client

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
            content = "\n\n".join(
                f"[{m['role'].upper()}] {m['content']}" for m in messages
            )
        elif prompt:
            content = prompt
        else:
            content = ""

        target_model = model or _DEFAULT_MODEL
        client = self._get_client()
        if not client:
            return {
                "model": target_model,
                "content": "[Gemini unavailable — check GEMINI_API_KEY]",
                "usage": {},
                "metadata": {"provider": "google", "error": "no_client"},
            }

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
        except Exception as exc:
            err_str = str(exc)
            # Quota exhausted — return graceful degradation instead of raising
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
                logger.warning("[gemini] quota exhausted: %s", err_str[:120])
                return {
                    "model": target_model,
                    "content": (
                        "[Ariston AI — Gemini quota exhausted for today's free tier. "
                        "The platform is operational; live AI analysis will resume when quota resets. "
                        "Set ANTHROPIC_API_KEY for immediate Claude-powered analysis.]"
                    ),
                    "usage": {},
                    "metadata": {"provider": "google", "error": "quota_exhausted"},
                }
            raise

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

