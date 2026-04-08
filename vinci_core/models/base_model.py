from abc import ABC, abstractmethod

class BaseModel(ABC):
    """
    Abstract base class for all Ariston AI models.
    Ensures consistent interface for generation and optional training/evaluation.
    """
    
    @abstractmethod
    async def generate(self, context: dict) -> dict:
        """Core generation interface used by the orchestrator."""
        return {"content": "", "model": "base"}

    # Machine Learning Lifecycle Methods (Optional)
    def train(self, data):
        """Optional training method."""
        pass

    def predict(self, input_data):
        """Optional prediction method."""
        pass

    def evaluate(self, test_data):
        """Optional evaluation method."""
        pass
