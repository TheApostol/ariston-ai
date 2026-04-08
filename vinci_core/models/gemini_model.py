import os
from google import genai
from config import settings
from vinci_core.models.base_model import BaseModel

class GeminiModel(BaseModel):
    def __init__(self, model_id: str = "gemini-1.5-pro"):
        self.model_id = model_id
        api_key = os.getenv("GEMINI_API_KEY") or settings.GEMINI_API_KEY
        self.client = genai.Client(api_key=api_key) if api_key else None

    async def generate(self, context: dict) -> dict:
        if not self.client:
            return {"content": "[Gemini Simulation] Missing API Key", "model": self.model_id}
            
        prompt = context.get("prompt", "")
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=prompt
        )
        return {"content": response.text, "model": self.model_id}
