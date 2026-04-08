import google.generativeai as genai
from config import settings

genai.configure(api_key=settings.GEMINI_API_KEY)


class GeminiModel:
    name = "gemini"

    def __init__(self):
        self.model = genai.GenerativeModel("gemini-1.5-pro")

    async def generate(self, context: dict) -> str:
        response = self.model.generate_content(context["prompt"])
        return response.text
