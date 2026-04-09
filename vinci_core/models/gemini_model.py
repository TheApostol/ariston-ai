import asyncio
from google import genai
from config import settings
from vinci_core.utils.retry import async_retry

_MODEL_NAME = "gemini-1.5-flash"


class GeminiModel:
    def __init__(self):
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)

    def _messages_to_text(self, messages: list) -> str:
        """Flatten a messages list into a single prompt string for Gemini."""
        parts = []
        for m in messages:
            role = m.get("role", "user").upper()
            content = m.get("content", "")
            if role == "SYSTEM":
                parts.append(f"[SYSTEM]\n{content}")
            else:
                parts.append(content)
        return "\n\n".join(parts)

    @async_retry(max_attempts=3, base_delay=1.0)
    async def generate(self, messages: list | None = None, prompt: str | None = None) -> dict:
        """
        Generate a response via Gemini.

        Accepts either:
          - messages: list of {"role": ..., "content": ...} dicts (standard interface)
          - prompt: raw string (legacy / convenience)
        """
        if messages:
            text_prompt = self._messages_to_text(messages)
        elif prompt:
            text_prompt = prompt
        else:
            raise ValueError("GeminiModel.generate requires `messages` or `prompt`")

        # Gemini SDK is synchronous — run in thread pool to keep the event loop free
        response = await asyncio.to_thread(
            self.client.models.generate_content,
            model=_MODEL_NAME,
            contents=text_prompt,
        )

        usage_meta = getattr(response, "usage_metadata", None)
        prompt_tokens = getattr(usage_meta, "prompt_token_count", 0) or 0
        completion_tokens = getattr(usage_meta, "candidates_token_count", 0) or 0

        return {
            "model": _MODEL_NAME,
            "content": response.text,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "metadata": {"provider": "google"},
        }
