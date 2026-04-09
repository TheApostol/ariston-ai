import os
import httpx
from vinci_core.utils.retry import async_retry


class OpenRouterModel:
    def __init__(self):
        self.api_key = None
        self.url = "https://openrouter.ai/api/v1/chat/completions"

    def get_api_key(self):
        if not self.api_key:
            self.api_key = os.getenv("OPENROUTER_API_KEY")
            if not self.api_key:
                raise ValueError("Missing OPENROUTER_API_KEY")
        return self.api_key

    @async_retry(max_attempts=3, base_delay=1.0)
    async def generate(self, messages=None, prompt=None):
        api_key = self.get_api_key()

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "Ariston AI",
        }

        payload = {
            "model": "openrouter/free",
            "messages": messages if messages else [
                {"role": "user", "content": prompt}
            ],
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(self.url, headers=headers, json=payload)

        try:
            data = response.json()
        except Exception:
            return {
                "model": "openrouter",
                "content": f"Invalid response: {response.text}",
                "usage": None,
                "metadata": {"error": True},
            }

        if "error" in data:
            return {
                "model": "openrouter",
                "content": str(data["error"]),
                "usage": None,
                "metadata": {"error": True},
            }

        return {
            "model": "openrouter",
            "content": data["choices"][0]["message"]["content"],
            "usage": data.get("usage"),
            "metadata": {"error": False},
        }
