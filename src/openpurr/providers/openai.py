"""OpenAI-compatible LLM provider.

Covers OpenAI, Gemini (via OpenAI compat endpoint), OpenRouter, DeepSeek,
llama.cpp and MLX behind the OpenAI Python SDK.
"""

from __future__ import annotations

from collections.abc import Generator

from openpurr.providers.base import BaseLLMProvider


class OpenAIError(RuntimeError):
    """Raised when the OpenAI-compatible API returns an error."""


def _rejects_temperature(exc: Exception) -> bool:
    """Check whether the exception signals a temperature rejection.

    Some reasoning-tier models (o1, o3, o4-mini and successors) reject any
    explicit ``temperature`` parameter instead of ignoring it.

    Args:
        exc: Exception raised by the OpenAI SDK.

    Returns:
        True if the error is about an unsupported ``temperature`` parameter,
        False otherwise.
    """
    body = getattr(exc, "body", None)
    if (
        isinstance(body, dict)
        and (body.get("error") or {}).get("param") == "temperature"
    ):
        return True
    text = str(exc).lower()
    return "temperature" in text and ("unsupported" in text or "not supported" in text)


class OpenAICompatibleProvider(BaseLLMProvider):
    """LLM provider using the OpenAI Python SDK for any compatible endpoint.

    Supports OpenAI, Gemini, OpenRouter, DeepSeek, llama.cpp and MLX. Gemini
    and others are accessed through their OpenAI-compatible ``/v1`` endpoints.

    Attributes:
        model: Model identifier to request.
    """

    def __init__(self, api_key: str, model: str, base_url: str | None = None) -> None:
        """Initialize the provider.

        Args:
            api_key: API key for the remote service. ``"not-needed"`` is
                substituted for local servers that require no auth.
            model: Model identifier.
            base_url: Optional base URL override (e.g. provider-specific
                endpoint). When ``None``, the SDK default is used.
        """
        self.model = model
        self._api_key = api_key
        self._base_url = base_url

    def _client(self):
        """Create an OpenAI client for the configured endpoint.

        Returns:
            An ``openai.OpenAI`` client instance.

        Raises:
            OpenAIError: If the ``openai`` package is not installed.
        """
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise OpenAIError(
                "openai package is not installed. Run: uv add openai"
            ) from exc
        return OpenAI(
            api_key=self._api_key or "not-needed", base_url=self._base_url or None
        )

    def _messages(self, prompt: str, system_prompt: str) -> list[dict[str, str]]:
        """Build the chat message list for the API call.

        Args:
            prompt: User prompt (git diff).
            system_prompt: System prompt.

        Returns:
            List of ``{"role": ..., "content": ...}`` messages.
        """
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

    def _complete(
        self, client, messages: list[dict[str, str]], temperature: float, stream: bool
    ):
        """Call the chat completions endpoint with temperature fallback.

        Retries without ``temperature`` if the model rejects the parameter.

        Args:
            client: OpenAI client instance.
            messages: Chat messages as produced by :meth:`_messages`.
            temperature: Sampling temperature.
            stream: Whether to request a streaming response.

        Returns:
            The SDK response object (or streaming iterable).
        """
        try:
            return client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=stream,
            )
        except Exception as exc:
            if not _rejects_temperature(exc):
                raise
            return client.chat.completions.create(
                model=self.model, messages=messages, stream=stream
            )

    def generate(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.0,
        keep_alive: str | None = None,
    ) -> str:
        """Generate a complete completion.

        Args:
            prompt: User prompt.
            system_prompt: System prompt.
            temperature: Sampling temperature.
            keep_alive: Ignored for this provider; present for interface
                compatibility.

        Returns:
            Generated text, or empty string if the choice has no content.

        Raises:
            OpenAIError: If the API call fails.
        """
        try:
            response = self._complete(
                self._client(),
                self._messages(prompt, system_prompt),
                temperature,
                stream=False,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            raise OpenAIError(str(exc)) from exc

    def generate_stream(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.0,
        keep_alive: str | None = None,
    ) -> Generator[str, None, None]:
        """Generate a streaming completion.

        Args:
            prompt: User prompt.
            system_prompt: System prompt.
            temperature: Sampling temperature.
            keep_alive: Ignored for this provider.

        Yields:
            Text deltas as they arrive from the model.

        Raises:
            OpenAIError: If the API call fails.
        """
        try:
            stream = self._complete(
                self._client(),
                self._messages(prompt, system_prompt),
                temperature,
                stream=True,
            )
            for chunk in stream:
                text = chunk.choices[0].delta.content
                if text:
                    yield text
        except Exception as exc:
            raise OpenAIError(str(exc)) from exc
