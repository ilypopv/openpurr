"""Ollama LLM provider.

Implements :class:`openpurr.providers.base.BaseLLMProvider` for a local
Ollama server via its ``/api/generate`` HTTP endpoint.
"""

from __future__ import annotations

import json
from collections.abc import Generator

import httpx

from openpurr.providers.base import BaseLLMProvider


class OllamaError(RuntimeError):
    """Raised when the Ollama API returns an error or is unreachable."""


class OllamaProvider(BaseLLMProvider):
    """LLM provider backed by a local Ollama server.

    Attributes:
        host: Base URL of the Ollama server, without trailing slash.
        model: Model name as known to Ollama (e.g. ``gemma3:27b``).
        timeout: HTTP timeout in seconds for generate calls.
    """

    def __init__(self, host: str, model: str, timeout: float = 300.0) -> None:
        """Initialize the provider.

        Args:
            host: Base URL of the Ollama server.
            model: Model name to query.
            timeout: HTTP timeout in seconds.
        """
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout

    def _payload(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float,
        keep_alive: str | None,
        stream: bool,
    ) -> dict:
        """Build the JSON payload for ``/api/generate``.

        Args:
            prompt: User prompt (git diff).
            system_prompt: System prompt.
            temperature: Sampling temperature.
            keep_alive: Ollama ``keep_alive`` hint.
            stream: Whether to request streaming.

        Returns:
            Dictionary suitable for json-encoding as the POST body.
        """
        return {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": stream,
            "keep_alive": keep_alive,
            "options": {"temperature": temperature},
            # Explicitly disable extended thinking: reasoning-capable models default
            # to emitting their chain-of-thought inline in "response" otherwise. A
            # no-op for models without thinking support.
            "think": False,
        }

    def generate(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.0,
        keep_alive: str | None = "5m",
    ) -> str:
        """Generate a complete completion via Ollama.

        Args:
            prompt: User prompt.
            system_prompt: System prompt.
            temperature: Sampling temperature.
            keep_alive: VRAM keep-alive duration (e.g. ``"5m"``, ``"0s"``).

        Returns:
            Generated text; empty string if the response has no ``response`` key.

        Raises:
            OllamaError: If the server is unreachable or returns an HTTP error.
        """
        payload = self._payload(
            prompt, system_prompt, temperature, keep_alive, stream=False
        )
        try:
            response = httpx.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.ConnectError as exc:
            raise OllamaError(
                f"Unable to connect to Ollama at {self.host}. Ensure the server is running."
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise OllamaError(f"Ollama API error: {exc}") from exc
        return response.json().get("response", "")

    def generate_stream(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.0,
        keep_alive: str | None = "5m",
    ) -> Generator[str, None, None]:
        """Generate a streaming completion via Ollama.

        Args:
            prompt: User prompt.
            system_prompt: System prompt.
            temperature: Sampling temperature.
            keep_alive: VRAM keep-alive duration.

        Yields:
            Text chunks as they arrive from the streaming endpoint.

        Raises:
            OllamaError: If the server is unreachable or returns an HTTP error.
        """
        payload = self._payload(
            prompt, system_prompt, temperature, keep_alive, stream=True
        )
        try:
            with httpx.stream(
                "POST",
                f"{self.host}/api/generate",
                json=payload,
                timeout=self.timeout,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    text = chunk.get("response", "")
                    if text:
                        yield text
                    if chunk.get("done"):
                        break
        except httpx.ConnectError as exc:
            raise OllamaError(
                f"Unable to connect to Ollama at {self.host}. Ensure the server is running."
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise OllamaError(f"Ollama API error: {exc}") from exc
