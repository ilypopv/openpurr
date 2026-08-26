"""Tests for OllamaProvider — mocks httpx to verify payloads and error handling."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from openpurr.providers.ollama import OllamaError, OllamaProvider

# ─── helpers ──────────────────────────────────────────────────────────────────


def _ok_response(text: str = "generated text") -> MagicMock:
    m = MagicMock()
    m.json.return_value = {"response": text, "done": True}
    m.raise_for_status = MagicMock()
    return m


def _stream_response(tokens: list[str]) -> MagicMock:
    lines = [json.dumps({"response": t, "done": False}) for t in tokens]
    lines.append(json.dumps({"response": "", "done": True}))
    m = MagicMock()
    m.__enter__ = lambda s: s
    m.__exit__ = MagicMock(return_value=False)
    m.iter_lines.return_value = iter(lines)
    m.raise_for_status = MagicMock()
    return m


# ─── generate() ───────────────────────────────────────────────────────────────


class TestOllamaProviderGenerate:
    def test_correct_model_in_payload(self):
        provider = OllamaProvider(host="http://localhost:11434", model="gemma4:26b")
        captured: dict = {}

        def fake_post(url, json, timeout):
            captured.update(json)
            return _ok_response()

        with patch("openpurr.providers.ollama.httpx.post", side_effect=fake_post):
            provider.generate("prompt", "system")

        assert captured["model"] == "gemma4:26b"

    def test_prompt_and_system_in_payload(self):
        provider = OllamaProvider(host="http://localhost:11434", model="m")
        captured: dict = {}

        def fake_post(url, json, timeout):
            captured.update(json)
            return _ok_response()

        with patch("openpurr.providers.ollama.httpx.post", side_effect=fake_post):
            provider.generate("my diff", "my system prompt")

        assert captured["prompt"] == "my diff"
        assert captured["system"] == "my system prompt"

    def test_stream_false_in_payload(self):
        provider = OllamaProvider(host="http://localhost:11434", model="m")
        captured: dict = {}

        def fake_post(url, json, timeout):
            captured.update(json)
            return _ok_response()

        with patch("openpurr.providers.ollama.httpx.post", side_effect=fake_post):
            provider.generate("p", "s")

        assert captured["stream"] is False

    def test_temperature_in_options(self):
        provider = OllamaProvider(host="http://localhost:11434", model="m")
        captured: dict = {}

        def fake_post(url, json, timeout):
            captured.update(json)
            return _ok_response()

        with patch("openpurr.providers.ollama.httpx.post", side_effect=fake_post):
            provider.generate("p", "s", temperature=0.0)

        assert captured["options"]["temperature"] == 0.0

    def test_keep_alive_passed_through(self):
        provider = OllamaProvider(host="http://localhost:11434", model="m")
        captured: dict = {}

        def fake_post(url, json, timeout):
            captured.update(json)
            return _ok_response()

        with patch("openpurr.providers.ollama.httpx.post", side_effect=fake_post):
            provider.generate("p", "s", keep_alive="5m")

        assert captured["keep_alive"] == "5m"

    def test_keep_alive_zero_unloads_vram(self):
        provider = OllamaProvider(host="http://localhost:11434", model="m")
        captured: dict = {}

        def fake_post(url, json, timeout):
            captured.update(json)
            return _ok_response()

        with patch("openpurr.providers.ollama.httpx.post", side_effect=fake_post):
            provider.generate("p", "s", keep_alive="0s")

        assert captured["keep_alive"] == "0s"

    def test_returns_response_text(self):
        provider = OllamaProvider(host="http://localhost:11434", model="m")
        with patch(
            "openpurr.providers.ollama.httpx.post",
            return_value=_ok_response("feat: add thing"),
        ):
            result = provider.generate("p", "s")
        assert result == "feat: add thing"

    def test_trailing_slash_stripped_from_host(self):
        provider = OllamaProvider(host="http://localhost:11434/", model="m")
        captured_url: list[str] = []

        def fake_post(url, **kwargs):
            captured_url.append(url)
            return _ok_response()

        with patch("openpurr.providers.ollama.httpx.post", side_effect=fake_post):
            provider.generate("p", "s")

        assert captured_url[0] == "http://localhost:11434/api/generate"

    def test_raises_on_connect_error(self):
        provider = OllamaProvider(host="http://localhost:11434", model="m")
        with (
            patch(
                "openpurr.providers.ollama.httpx.post",
                side_effect=httpx.ConnectError("connection refused"),
            ),
            pytest.raises(OllamaError, match="Unable to connect"),
        ):
            provider.generate("p", "s")

    def test_raises_on_http_status_error(self):
        provider = OllamaProvider(host="http://localhost:11434", model="m")
        resp = MagicMock()
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "404", request=MagicMock(), response=MagicMock()
        )
        with (
            patch("openpurr.providers.ollama.httpx.post", return_value=resp),
            pytest.raises(OllamaError, match="API error"),
        ):
            provider.generate("p", "s")


# ─── generate_stream() ────────────────────────────────────────────────────────


class TestOllamaProviderStream:
    def test_yields_tokens(self):
        provider = OllamaProvider(host="http://localhost:11434", model="m")
        with patch(
            "openpurr.providers.ollama.httpx.stream",
            return_value=_stream_response(["hello", " world"]),
        ):
            result = list(provider.generate_stream("p", "s"))
        assert result == ["hello", " world"]

    def test_empty_tokens_not_yielded(self):
        provider = OllamaProvider(host="http://localhost:11434", model="m")
        with patch(
            "openpurr.providers.ollama.httpx.stream",
            return_value=_stream_response(["token"]),
        ):
            result = list(provider.generate_stream("p", "s"))
        # The final done=True line has response="" which should not be yielded
        assert "" not in result

    def test_stream_true_in_payload(self):
        provider = OllamaProvider(host="http://localhost:11434", model="m")
        captured: dict = {}

        def fake_stream(method, url, json, timeout):
            captured.update(json)
            return _stream_response([])

        with patch("openpurr.providers.ollama.httpx.stream", side_effect=fake_stream):
            list(provider.generate_stream("p", "s"))

        assert captured["stream"] is True

    def test_raises_on_connect_error(self):
        provider = OllamaProvider(host="http://localhost:11434", model="m")
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(side_effect=httpx.ConnectError("refused"))
        mock_ctx.__exit__ = MagicMock(return_value=False)
        with (
            patch("openpurr.providers.ollama.httpx.stream", return_value=mock_ctx),
            pytest.raises(OllamaError, match="Unable to connect"),
        ):
            list(provider.generate_stream("p", "s"))
