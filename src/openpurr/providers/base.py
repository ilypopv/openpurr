"""Abstract base class for LLM providers.

Defines the common interface that all openpurr providers must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Generator


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers.

    Subclasses must implement text generation via both non-streaming and
    streaming APIs while honoring the same parameter contract.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.0,
        keep_alive: str | None = "5m",
    ) -> str:
        """Generate a complete completion.

        Args:
            prompt: User prompt, typically a git diff.
            system_prompt: System prompt that controls output style.
            temperature: Sampling temperature. 0.0 is deterministic.
            keep_alive: Ollama VRAM keep-alive hint. Ignored by other
                providers. ``None`` lets the server decide.

        Returns:
            Generated text stripped of any provider envelope.

        Raises:
            NotImplementedError: If the subclass does not implement the method.
        """
        raise NotImplementedError

    @abstractmethod
    def generate_stream(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.0,
        keep_alive: str | None = "5m",
    ) -> Generator[str, None, None]:
        """Generate a streaming completion.

        Args:
            prompt: User prompt, typically a git diff.
            system_prompt: System prompt that controls output style.
            temperature: Sampling temperature. 0.0 is deterministic.
            keep_alive: Ollama VRAM keep-alive hint. Ignored by other
                providers.

        Yields:
            Successive text chunks as produced by the underlying model.

        Raises:
            NotImplementedError: If the subclass does not implement the method.
        """
        raise NotImplementedError
