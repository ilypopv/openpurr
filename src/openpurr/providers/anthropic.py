"""Anthropic Claude LLM provider."""

from __future__ import annotations

from collections.abc import Generator

from openpurr.providers.base import BaseLLMProvider


class AnthropicError(RuntimeError):
    """Raised when the Anthropic API returns an error."""


class AnthropicProvider(BaseLLMProvider):
    """LLM provider for Anthropic Claude models.

    Attributes:
        model: Model identifier (e.g. ``claude-sonnet-4-0``).
    """

    def __init__(self, api_key: str, model: str) -> None:
        """Initialize the provider.

        Args:
            api_key: Anthropic API key.
            model: Model identifier to request.
        """
        self.model = model
        self._api_key = api_key

    def _client(self):
        """Create an Anthropic client.

        Returns:
            An ``anthropic.Anthropic`` client instance.

        Raises:
            AnthropicError: If the ``anthropic`` package is not installed.
        """
        try:
            import anthropic
        except ImportError as exc:
            raise AnthropicError(
                "anthropic package is not installed. Run: uv add anthropic"
            ) from exc
        return anthropic.Anthropic(api_key=self._api_key)

    def generate(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.0,
        keep_alive: str | None = None,
    ) -> str:
        """Generate a complete completion via Anthropic.

        Args:
            prompt: User prompt.
            system_prompt: System prompt.
            temperature: Sampling temperature.
            keep_alive: Ignored for this provider; present for interface
                compatibility.

        Returns:
            Concatenated text from the response content blocks.

        Raises:
            AnthropicError: If the API call fails.
        """
        try:
            message = self._client().messages.create(
                model=self.model,
                max_tokens=4096,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            return "".join(
                block.text for block in message.content if block.type == "text"
            )
        except Exception as exc:
            raise AnthropicError(str(exc)) from exc

    def generate_stream(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.0,
        keep_alive: str | None = None,
    ) -> Generator[str, None, None]:
        """Generate a streaming completion via Anthropic.

        Args:
            prompt: User prompt.
            system_prompt: System prompt.
            temperature: Sampling temperature.
            keep_alive: Ignored for this provider.

        Yields:
            Text chunks from the streaming response.

        Raises:
            AnthropicError: If the API call fails.
        """
        try:
            with self._client().messages.stream(
                model=self.model,
                max_tokens=4096,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            ) as stream:
                yield from stream.text_stream
        except Exception as exc:
            raise AnthropicError(str(exc)) from exc
