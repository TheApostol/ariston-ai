"""
OpenAI provider — normalized interface with async execution, timeout, and vision support.

Returns: { model, content, usage, metadata }

Gracefully degrades when OPENAI_API_KEY is absent or 401/429 errors occur.
Supports vision: pass images=[<base64_or_url>, ...] to encode as OpenAI vision messages.
"""

import asyncio
import base64
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ariston.providers.openai")

_DEFAULT_MODEL = "gpt-4o-mini"
_TIMEOUT_SECONDS = 60


class OpenAIModel:
    name = "openai"
    DEFAULT_MODEL = _DEFAULT_MODEL

    def __init__(self):
        self._client = None  # lazy-initialized

    def _get_client(self):
        if self._client is not None:
            return self._client

        # Try config settings first, fall back to env
        api_key = None
        try:
            from config import settings
            api_key = getattr(settings, "OPENAI_API_KEY", None)
        except Exception:
            pass
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            logger.warning("[openai] No OPENAI_API_KEY configured — OpenAI provider disabled")
            return None

        try:
            import openai  # lazy import — SDK optional
            self._client = openai.AsyncOpenAI(
                api_key=api_key,
                timeout=_TIMEOUT_SECONDS,
            )
        except (ImportError, Exception) as e:
            logger.warning("[openai] client init failed: %s", e)

        return self._client

    @staticmethod
    def _build_vision_messages(
        messages: List[Dict], images: List[str]
    ) -> List[Dict]:
        """
        Encode images as OpenAI vision message content blocks.
        Each image may be a URL (http/https) or a base64-encoded string.
        The images are appended to the last user message, or a new user message is created.
        """
        image_content = []
        for img in images:
            if img.startswith("http://") or img.startswith("https://"):
                image_content.append({"type": "image_url", "image_url": {"url": img}})
            else:
                # Assume raw base64; wrap in data URI
                image_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img}"},
                })

        # Append to last user message or create one
        new_messages = list(messages)
        last_user_idx = None
        for i in range(len(new_messages) - 1, -1, -1):
            if new_messages[i].get("role") == "user":
                last_user_idx = i
                break

        if last_user_idx is not None:
            existing = new_messages[last_user_idx]
            content = existing.get("content", "")
            text_block = {"type": "text", "text": content} if isinstance(content, str) else content
            new_messages[last_user_idx] = {
                "role": "user",
                "content": ([text_block] if isinstance(text_block, dict) else text_block)
                + image_content,
            }
        else:
            new_messages.append({"role": "user", "content": image_content})

        return new_messages

    async def generate(
        self,
        messages: Optional[List[Dict[str, str]]] = None,
        prompt: Optional[str] = None,
        model: Optional[str] = None,
        images: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Generate a completion via OpenAI.

        Args:
            messages: List of {role, content} dicts.
            prompt:   Convenience shorthand — wrapped as a user message.
            model:    Override model (default: gpt-4o-mini).
            images:   List of image URLs or base64 strings for vision tasks.

        Returns normalized {model, content, usage, metadata} dict.
        """
        client = self._get_client()
        used_model = model or _DEFAULT_MODEL

        if not client:
            return {
                "model": used_model,
                "content": "[OpenAI unavailable — check OPENAI_API_KEY]",
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "metadata": {"provider": "openai", "error": "no_client"},
            }

        # Build message list
        if messages:
            msg_list = list(messages)
        elif prompt:
            msg_list = [{"role": "user", "content": prompt}]
        else:
            msg_list = []

        # Inject vision content if images provided
        if images:
            msg_list = self._build_vision_messages(msg_list, images)

        logger.debug("[openai] calling model=%s messages=%d", used_model, len(msg_list))

        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=used_model,
                    messages=msg_list,
                    max_tokens=kwargs.get("max_tokens", 2048),
                    temperature=kwargs.get("temperature", 0.7),
                ),
                timeout=_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.error("[openai] timeout after %ds", _TIMEOUT_SECONDS)
            return {
                "model": used_model,
                "content": "[OpenAI request timed out]",
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "metadata": {"provider": "openai", "error": "timeout"},
            }
        except Exception as exc:
            err_str = str(exc)
            # 401 Unauthorized — bad key
            if "401" in err_str or "Unauthorized" in err_str or "invalid_api_key" in err_str:
                logger.warning("[openai] invalid API key: %s", err_str[:120])
                return {
                    "model": used_model,
                    "content": "[OpenAI authentication failed — check OPENAI_API_KEY]",
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    "metadata": {"provider": "openai", "error": "auth_failed"},
                }
            # 429 Rate limit / quota
            if "429" in err_str or "rate_limit" in err_str.lower() or "quota" in err_str.lower():
                logger.warning("[openai] rate limit / quota: %s", err_str[:120])
                return {
                    "model": used_model,
                    "content": (
                        "[Ariston AI — OpenAI rate limit reached. "
                        "The platform is operational; retrying shortly.]"
                    ),
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    "metadata": {"provider": "openai", "error": "rate_limit"},
                }
            # Other errors — re-raise so the router can fall through its chain
            raise

        choice = response.choices[0]
        usage = response.usage

        return {
            "model": response.model,
            "content": choice.message.content or "",
            "usage": {
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0,
            },
            "metadata": {
                "provider": "openai",
                "finish_reason": choice.finish_reason,
            },
        }
