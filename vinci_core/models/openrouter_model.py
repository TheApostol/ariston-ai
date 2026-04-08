import httpx
from config import settings


class OpenRouterModel:
    async def generate(self, context: dict) -> dict:
        prompt = context.get("prompt", "")

        if not settings.OPENROUTER_API_KEY:
            raise ValueError("Missing OPENROUTER_API_KEY")

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "openai/gpt-3.5-turbo",
                    "messages": [
                        {"role": "user", "content": prompt}
                    ]
                }
            )

            if response.status_code != 200:
                raise Exception(response.text)

            return response.json()
