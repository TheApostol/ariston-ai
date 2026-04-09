import httpx


class OllamaModel:
    async def generate(self, prompt: str):
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3:8b",
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 200,
                    },
                    "keep_alive": "10m",
                },
            )

        data = response.json()

        return {
            "model": "ollama",
            "content": data.get("response", ""),
            "usage": None,
            "metadata": {"error": False},
        }