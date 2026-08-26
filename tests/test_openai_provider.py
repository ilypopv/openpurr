"""Tests for OpenAICompatibleProvider — mocks the openai SDK boundary.

Focus: the retry-without-temperature fallback for reasoning-tier models
(o1/o3/o4-mini and similar) that reject an explicit `temperature` value.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from openpurr.providers.openai import (
    OpenAICompatibleProvider,
    OpenAIError,
    _rejects_temperature,
)

# ─── helpers ──────────────────────────────────────────────────────────────────


def _response(text: str = "generated text") -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content=text))]
    return resp


def _chunks(tokens: list[str]):
    for t in tokens:
        yield MagicMock(choices=[MagicMock(delta=MagicMock(content=t))])


def _temperature_rejected_error() -> RuntimeError:
    exc = RuntimeError(
        "Unsupported value: 'temperature' does not support 0.0 with this model."
    )
    exc.body = {"error": {"param": "temperature", "code": "unsupported_value"}}
    return exc


# ─── _rejects_temperature heuristic ──────────────────────────────────────────


class TestRejectsTemperatureHeuristic:
    def test_detects_via_body_param(self):
        exc = RuntimeError("boom")
        exc.body = {"error": {"param": "temperature"}}
        assert _rejects_temperature(exc)

    def test_detects_via_message_text(self):
        exc = RuntimeError(
            "Unsupported parameter: 'temperature' is not supported with this model."
        )
        assert _rejects_temperature(exc)

    def test_unrelated_error_not_detected(self):
        assert not _rejects_temperature(RuntimeError("connection refused"))


# ─── generate() ───────────────────────────────────────────────────────────────


class TestGenerate:
    def test_temperature_passed_by_default(self):
        provider = OpenAICompatibleProvider(api_key="k", model="gpt-4o")
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = _response()
        with patch("openai.OpenAI", return_value=fake_client):
            provider.generate("prompt", "system", temperature=0.5)
        _, kwargs = fake_client.chat.completions.create.call_args
        assert kwargs["temperature"] == 0.5

    def test_retries_without_temperature_when_rejected(self):
        provider = OpenAICompatibleProvider(api_key="k", model="o3")
        fake_client = MagicMock()
        fake_client.chat.completions.create.side_effect = [
            _temperature_rejected_error(),
            _response("ok"),
        ]
        with patch("openai.OpenAI", return_value=fake_client):
            result = provider.generate("prompt", "system", temperature=0.0)
        assert result == "ok"
        assert fake_client.chat.completions.create.call_count == 2
        _, retry_kwargs = fake_client.chat.completions.create.call_args
        assert "temperature" not in retry_kwargs

    def test_does_not_retry_on_unrelated_error(self):
        provider = OpenAICompatibleProvider(api_key="k", model="gpt-4o")
        fake_client = MagicMock()
        fake_client.chat.completions.create.side_effect = RuntimeError("rate limited")
        with (
            patch("openai.OpenAI", return_value=fake_client),
            pytest.raises(OpenAIError, match="rate limited"),
        ):
            provider.generate("prompt", "system")
        assert fake_client.chat.completions.create.call_count == 1

    def test_raises_openai_error_if_retry_also_fails(self):
        provider = OpenAICompatibleProvider(api_key="k", model="o3")
        fake_client = MagicMock()
        fake_client.chat.completions.create.side_effect = [
            _temperature_rejected_error(),
            RuntimeError("still broken"),
        ]
        with (
            patch("openai.OpenAI", return_value=fake_client),
            pytest.raises(OpenAIError, match="still broken"),
        ):
            provider.generate("prompt", "system")
        assert fake_client.chat.completions.create.call_count == 2


# ─── generate_stream() ────────────────────────────────────────────────────────


class TestGenerateStream:
    def test_retries_without_temperature_when_rejected(self):
        provider = OpenAICompatibleProvider(api_key="k", model="o3")
        fake_client = MagicMock()
        fake_client.chat.completions.create.side_effect = [
            _temperature_rejected_error(),
            _chunks(["hello", " world"]),
        ]
        with patch("openai.OpenAI", return_value=fake_client):
            result = list(provider.generate_stream("prompt", "system", temperature=0.0))
        assert result == ["hello", " world"]
        assert fake_client.chat.completions.create.call_count == 2
        _, retry_kwargs = fake_client.chat.completions.create.call_args
        assert "temperature" not in retry_kwargs

    def test_does_not_retry_on_unrelated_error(self):
        provider = OpenAICompatibleProvider(api_key="k", model="gpt-4o")
        fake_client = MagicMock()
        fake_client.chat.completions.create.side_effect = RuntimeError("boom")
        with (
            patch("openai.OpenAI", return_value=fake_client),
            pytest.raises(OpenAIError, match="boom"),
        ):
            list(provider.generate_stream("prompt", "system"))
        assert fake_client.chat.completions.create.call_count == 1
