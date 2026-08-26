"""OpenAI-compatible LLM provider (OpenAI, OpenRouter, DeepSeek, llama.cpp, MLX)."""

from __future__ import annotations

from collections.abc import Generator

from openpurr.providers.base import BaseLLMProvider


class OpenAIError(RuntimeError):
    """Raised when the OpenAI-compatible API returns an error."""


def _rejects_temperature(exc: Exception) -> bool:
    """True if `exc` is the API refusing a non-default `temperature`.

    Reasoning-tier models (o1, o3, o4-mini, and similar future releases)
    error on any explicit `temperature` value instead of silently ignoring it.
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

    def _messages(self, prompt: str, system_prompt: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

    def _complete(
        self, client, messages: list[dict[str, str]], temperature: float, stream: bool
    ):
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
