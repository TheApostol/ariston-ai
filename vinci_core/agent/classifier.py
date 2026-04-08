from vinci_core.models.openrouter_model import OpenRouterModel

class IntentClassifier:
    """
    Autonomous layer that acts as the core 'Vinci AI' brain. 
    Before executing a workflow, it reasons about the user intent to self-route the request.
    """
    def __init__(self):
        self.brain = OpenRouterModel()

    async def classify(self, prompt: str) -> str:
        # Heuristic 1: Vision keywords
        prompt_lower = prompt.lower()
        if any(w in prompt_lower for w in ["x-ray", "mri", "image", "scan", "radiograph"]):
            return "radiology"
            
        system_prompt = (
            "You are the root orchestrator of a medical AI system. Classify the user's prompt into exactly one of these categories:\n"
            "1. 'clinical' - symptoms, diagnosis, patient care, diseases.\n"
            "2. 'pharma' - drug queries, interactions, pharmacology.\n"
            "3. 'data' - parsing structured lists, extracting entities, non-reasoning data jobs.\n"
            "4. 'general' - greetings, normal chat, unrelated to medicine.\n\n"
            "Reply with ONLY the exact category string (e.g. 'pharma'). Do NOT explain.\n\n"
            f"Prompt: {prompt}"
        )
        try:
            # Bypass engine safety checks since this is internal routing
            result = await self.brain.generate({"prompt": system_prompt})
            
            # Extract content manually
            content = ""
            if isinstance(result, dict):
                if "choices" in result:
                    content = result["choices"][0]["message"]["content"]
                elif "candidates" in result:
                    content = result["candidates"][0]["content"]["parts"][0]["text"]
                else:
                    content = str(result)
            else:
                content = str(result)
                
            content = content.lower().strip()
            
            # Fuzzy match layer
            valid_layers = ["clinical", "pharma", "data", "general"]
            for layer in valid_layers:
                if layer in content:
                    return layer
            return "general"
        except Exception as e:
            print(f"Classifier brain failed: {e}")
            return "general"
