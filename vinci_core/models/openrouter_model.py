import httpx
from config import settings
from vinci_core.models.base_model import BaseModel

class OpenRouterModel(BaseModel):
    def __init__(self, model_id: str = "openai/gpt-3.5-turbo"):
        self.model_id = model_id

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
                    "model": self.model_id,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ]
                }
            )

            if response.status_code != 200:
                raise Exception(response.text)

            result = response.json()
            # Normalize extract for consensus model
            content = ""
            if "choices" in result:
                content = result["choices"][0]["message"]["content"]
            else:
                content = str(result)
            
            return {"content": content, "model": self.model_id, "raw": result}
