import httpx
from typing import Any
from vinci_core.models.base_model import BaseModel

class OllamaModel(BaseModel):
    """
    HIPAA-compliant local inference model connecting to an air-gapped instances of Ollama.
    """
    def __init__(self, model_name: str = "llama3"):
        self.model_name = model_name

    async def generate(self, context: dict) -> dict:
        prompt = context.get("prompt", "")
        url = "http://localhost:11434/api/generate"
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.post(url, json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False
                })
                response.raise_for_status()
                data = response.json()
                return {"content": data.get("response", "")}
            except Exception as e:
                raise Exception(f"Ollama execution failed: {str(e)}")

    def train(self, data: Any):
        """Not implemented for local inference proxy"""
        pass

    def predict(self, input_data: Any) -> Any:
        """Not implemented for local inference proxy"""
        pass
