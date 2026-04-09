import httpx
from vinci_core.utils.retry import async_retry

_OLLAMA_URL = "http://localhost:11434/api/generate"
_MODEL_NAME = "llama3:8b"


class OllamaModel:
    def _messages_to_prompt(self, messages: list) -> str:
        """Flatten a messages list into a plain-text prompt for Ollama."""
        parts = []
        for m in messages:
            role = m.get("role", "user").upper()
            content = m.get("content", "")
            if role == "SYSTEM":
                parts.append(f"[SYSTEM]\n{content}")
            else:
                parts.append(content)
        return "\n\n".join(parts)

    @async_retry(max_attempts=2, base_delay=0.5)
    async def generate(self, messages: list | None = None, prompt: str | None = None) -> dict:
        """
        Generate via a locally-running Ollama instance.

        Accepts either:
          - messages: list of {"role": ..., "content": ...} dicts (standard interface)
          - prompt: raw string (legacy / convenience)
        """
        if messages:
            text_prompt = self._messages_to_prompt(messages)
        elif prompt:
            text_prompt = prompt
        else:
            raise ValueError("OllamaModel.generate requires `messages` or `prompt`")

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                _OLLAMA_URL,
                json={
                    "model": _MODEL_NAME,
                    "prompt": text_prompt,
                    "stream": False,
                    "options": {"temperature": 0.7, "num_predict": 200},
                    "keep_alive": "10m",
                },
            )
        data = response.json()

        return {
            "model": _MODEL_NAME,
            "content": data.get("response", ""),
            "usage": {
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
                "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
            },
            "metadata": {"provider": "ollama"},
        }