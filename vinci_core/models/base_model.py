"""
Abstract base class for all Ariston AI model providers.

Every provider MUST return a dict with this exact shape:
    {
        "model":    str,           # model identifier string
        "content":  str,           # generated text
        "usage":    dict | None,   # token usage (prompt/completion/total)
        "metadata": dict,          # provider-specific extras (provider, latency_ms, …)
    }

The `generate` method accepts keyword-only arguments so callers can
pass `messages` or `prompt` interchangeably.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseModel(ABC):
    """Abstract base for all Ariston AI model providers."""

    #: Human-readable provider name, used in metadata.
    name: str = "base"

    @abstractmethod
    async def generate(
        self,
        messages: Optional[List[Dict[str, str]]] = None,
        prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Generate a completion.

        Implementors must accept at minimum either *messages* (list of
        ``{"role": ..., "content": ...}`` dicts) or a plain *prompt* string,
        and return the normalised response dict.

        Returns:
            {
                "model":    str,
                "content":  str,
                "usage":    {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int} | None,
                "metadata": {"provider": str, ...},
            }
        """
        ...

    # ── Optional ML lifecycle hooks (no-op by default) ────────────────────────

    def train(self, data: Any) -> None:
        """Optional training hook."""

    def predict(self, input_data: Any) -> Any:
        """Optional prediction hook."""

    def evaluate(self, test_data: Any) -> Any:
        """Optional evaluation hook."""

