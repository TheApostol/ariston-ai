import anthropic
from config import settings
from vinci_core.utils.retry import async_retry


class AnthropicModel:
    name = "anthropic"

    def __init__(self):
        self.client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)

    @async_retry(max_attempts=3, base_delay=1.0)
    async def generate(self, messages: list) -> dict:
        # Strip system messages — Anthropic uses a separate system param
        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        user_messages = [m for m in messages if m["role"] != "system"]
        system_prompt = "\n\n".join(system_parts) if system_parts else None

        kwargs = {
            "model": "claude-sonnet-4-6",
            "max_tokens": 2048,
            "messages": user_messages,
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        response = await self.client.messages.create(**kwargs)

        return {
            "model": "claude-sonnet-4-6",
            "content": response.content[0].text,
            "usage": {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            },
            "metadata": {"provider": "anthropic"},
        }
