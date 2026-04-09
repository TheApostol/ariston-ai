import openai
from config import settings
from vinci_core.models.base_model import BaseModel

class OpenAIModel(BaseModel):
    def __init__(self, model_id: str = "gpt-4"):
        self.model_id = model_id
        self.client = openai.AsyncOpenAI(api_key=settings.openai_api_key)

    async def generate(self, context: dict) -> dict:
        prompt = context.get("prompt", "")
        response = await self.client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
        )
        return {"content": response.choices[0].message.content, "model": self.model_id}
