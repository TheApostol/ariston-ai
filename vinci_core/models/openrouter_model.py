import logging
import os
import httpx

logger = logging.getLogger("ariston.openrouter")

_TIMEOUT_SECONDS = 45.0
_FREE_MODEL = "meta-llama/llama-3.3-8b-instruct:free"


class OpenRouterModel:
    """
    OpenRouter model — normalized response shape matching all Ariston providers.
    Returns: {model, content, usage, metadata}
    """

    def __init__(self):
        self._api_key: str | None = None
        self.url = "https://openrouter.ai/api/v1/chat/completions"

    @property
    def api_key(self) -> str:
        if not self._api_key:
            key = os.getenv("OPENROUTER_API_KEY")
            if not key:
                raise ValueError("Missing OPENROUTER_API_KEY environment variable")
            self._api_key = key
        return self._api_key

    async def generate(self, messages: list = None, prompt: str = None) -> dict:
        if not messages and not prompt:
            raise ValueError("Either messages or prompt must be provided")

        msg_list = messages if messages else [{"role": "user", "content": prompt}]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://ariston.ai",
            "X-Title": "Ariston AI",
        }

        payload = {"model": _FREE_MODEL, "messages": msg_list}

        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
                response = await client.post(self.url, headers=headers, json=payload)
            data = response.json()
        except httpx.TimeoutException as e:
            logger.error("[OpenRouter] timeout: %s", e)
            raise TimeoutError(f"OpenRouter timed out: {e}")
        except Exception as e:
            logger.error("[OpenRouter] request failed: %s", e)
            raise

        if "error" in data:
            err = data["error"]
            logger.warning("[OpenRouter] API error: %s", err)
            raise RuntimeError(f"OpenRouter error: {err}")

        choice = data["choices"][0]["message"]["content"]
        raw_usage = data.get("usage") or {}

        return {
            "model": data.get("model", _FREE_MODEL),
            "content": choice,
            "usage": {
                "prompt_tokens": raw_usage.get("prompt_tokens", 0),
                "completion_tokens": raw_usage.get("completion_tokens", 0),
                "total_tokens": raw_usage.get("total_tokens", 0),
            },
            "metadata": {
                "provider": "openrouter",
                "fallback_used": False,
            },
        }
