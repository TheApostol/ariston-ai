import anthropic
from config import settings
from vinci_core.models.base_model import BaseModel

class AnthropicModel(BaseModel):
    def __init__(self, model_id: str = "claude-3-opus-20240229"):
        self.model_id = model_id
        self.client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

    async def generate(self, context: dict) -> dict:
        prompt = context.get("prompt", "")
        response = await self.client.messages.create(
            model=self.model_id,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return {"content": response.content[0].text, "model": self.model_id}
