"""OpenAI-compatible LLM provider (OpenAI, OpenRouter, DeepSeek, llama.cpp, MLX)."""

from __future__ import annotations

from collections.abc import Generator

from openpurr.providers.base import BaseLLMProvider


class OpenAIError(RuntimeError):
    """Raised when the OpenAI-compatible API returns an error."""


class OpenAICompatibleProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model: str, base_url: str | None = None) -> None:
        self.model = model
        self._api_key = api_key
        self._base_url = base_url

    def _client(self):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise OpenAIError(
                "openai package is not installed. Run: uv add openai"
            ) from exc
        return OpenAI(
            api_key=self._api_key or "not-needed", base_url=self._base_url or None
        )

    def generate(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.0,
        keep_alive: str | None = None,
    ) -> str:
        try:
            response = self._client().chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
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
        try:
            stream = self._client().chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                stream=True,
            )
            for chunk in stream:
                text = chunk.choices[0].delta.content
                if text:
                    yield text
        except Exception as exc:
            raise OpenAIError(str(exc)) from exc
