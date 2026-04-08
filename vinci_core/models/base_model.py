from abc import ABC, abstractmethod

class BaseModel(ABC):
    """
    Abstract base class for all AI models.
    """

    @abstractmethod
    def train(self, data):
        """Train the model with the given data."""
        pass

    @abstractmethod
    def predict(self, input_data):
        """Make predictions using the trained model."""
        pass

    @abstractmethod
    def evaluate(self, test_data):
        """Evaluate the model's performance on the test data."""
        pass
