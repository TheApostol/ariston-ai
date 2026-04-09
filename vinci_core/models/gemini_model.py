from google import genai
from config import settings


class GeminiModel:
    def __init__(self):
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)

    async def generate(self, prompt: str, context: dict | None = None):
        response = self.client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt,
        )

        return {
            "model": "gemini-1.5-flash",
            "content": response.text,
            "usage": {},
            "metadata": {"provider": "google"}
        }
