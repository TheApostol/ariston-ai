# vinci_core/models/anthropic_model.py

import anthropic
from config import settings

client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)


class AnthropicModel:
    name = "anthropic"

    async def generate(self, context: dict) -> str:
        response = await client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1024,
            messages=[{"role": "user", "content": context["prompt"]}],
        )
        return response.content[0].text
