"""Ollama LLM provider."""

from __future__ import annotations

import json
from collections.abc import Generator

import httpx
from openpurr.providers.base import BaseLLMProvider


class OllamaError(RuntimeError):
    """Raised when the Ollama API returns an error or is unreachable."""


class OllamaProvider(BaseLLMProvider):
    def __init__(self, host: str, model: str, timeout: float = 300.0) -> None:
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
        return {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": stream,
            "keep_alive": keep_alive,
            "options": {"temperature": temperature},
        }

    def generate(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.0,
        keep_alive: str | None = "5m",
    ) -> str:
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
