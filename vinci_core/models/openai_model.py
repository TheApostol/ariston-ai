# vinci_core/models/openai_model.py

from openai import AsyncOpenAI
from config import settings

client = AsyncOpenAI(api_key=settings.openai_api_key)


class OpenAIModel:
    name = "openai"

    async def generate(self, context: dict) -> str:
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": context["prompt"]}],
        )
        return response.choices[0].message.content
