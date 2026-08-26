"""Tests for openpurr.model_catalog — live model discovery per provider.

Every branch is mocked at the SDK/httpx boundary; the point of these tests is
that (a) results are sorted newest-first when a timestamp is available, (b)
non-chat OpenAI models are filtered out, and (c) failures never raise.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx

from openpurr import model_catalog

# ─── ollama ─────────────────────────────────────────────────────────────────────


class TestListModelsOllama:
    def test_parses_tags_response(self):
        resp = MagicMock()
        resp.json.return_value = {
            "models": [{"name": "llama3:8b"}, {"name": "gemma:2b"}]
        }
        resp.raise_for_status = MagicMock()
        with patch("openpurr.model_catalog.httpx.get", return_value=resp) as mock_get:
            names = model_catalog.list_models("ollama", host="http://localhost:11434")
        assert names == ["llama3:8b", "gemma:2b"]
        mock_get.assert_called_once()
        assert "api/tags" in mock_get.call_args[0][0]

    def test_returns_empty_list_on_connection_error(self):
        with patch(
            "openpurr.model_catalog.httpx.get",
            side_effect=httpx.ConnectError("refused"),
        ):
            assert (
                model_catalog.list_models("ollama", host="http://localhost:11434") == []
            )

    def test_defaults_host_when_none(self):
        resp = MagicMock()
        resp.json.return_value = {"models": []}
        resp.raise_for_status = MagicMock()
        with patch("openpurr.model_catalog.httpx.get", return_value=resp) as mock_get:
            model_catalog.list_models("ollama", host=None)
        assert "localhost:11434" in mock_get.call_args[0][0]


# ─── openai-compatible ──────────────────────────────────────────────────────────


def _model(id_: str, created: int) -> SimpleNamespace:
    return SimpleNamespace(id=id_, created=created)


class TestListModelsOpenAICompatible:
    def test_sorted_newest_first(self):
        models = [_model("gpt-a", 100), _model("gpt-b", 200)]
        fake_client = MagicMock()
        fake_client.models.list.return_value = models
        with patch("openai.OpenAI", return_value=fake_client):
            names = model_catalog.list_models("deepseek", api_key="k")
        assert names == ["gpt-b", "gpt-a"]

    def test_openai_filters_non_chat_models(self):
        models = [
            _model("gpt-4o", 100),
            _model("text-embedding-3-small", 200),
            _model("whisper-1", 150),
            _model("dall-e-3", 120),
        ]
        fake_client = MagicMock()
        fake_client.models.list.return_value = models
        with patch("openai.OpenAI", return_value=fake_client):
            names = model_catalog.list_models("openai", api_key="k")
        assert names == ["gpt-4o"]

    def test_non_openai_providers_not_filtered(self):
        models = [_model("deepseek-chat", 100), _model("deepseek-reasoner", 200)]
        fake_client = MagicMock()
        fake_client.models.list.return_value = models
        with patch("openai.OpenAI", return_value=fake_client):
            names = model_catalog.list_models("deepseek", api_key="k")
        assert set(names) == {"deepseek-chat", "deepseek-reasoner"}

    def test_returns_empty_on_exception(self):
        with patch("openai.OpenAI", side_effect=RuntimeError("boom")):
            assert model_catalog.list_models("openai", api_key="bad") == []

    def test_llamacpp_unreachable_returns_empty(self):
        fake_client = MagicMock()
        fake_client.models.list.side_effect = httpx.ConnectError("refused")
        with patch("openai.OpenAI", return_value=fake_client):
            assert model_catalog.list_models("llamacpp") == []


# ─── anthropic ──────────────────────────────────────────────────────────────────


class TestListModelsAnthropic:
    def test_sorted_by_created_at_desc(self):
        older = SimpleNamespace(id="claude-old", created_at="2024-01-01")
        newer = SimpleNamespace(id="claude-new", created_at="2025-01-01")
        fake_client = MagicMock()
        fake_client.models.list.return_value = [older, newer]
        with patch("anthropic.Anthropic", return_value=fake_client):
            names = model_catalog.list_models("anthropic", api_key="k")
        assert names == ["claude-new", "claude-old"]

    def test_returns_empty_on_exception(self):
        with patch("anthropic.Anthropic", side_effect=RuntimeError("boom")):
            assert model_catalog.list_models("anthropic", api_key="bad") == []


# ─── openrouter ─────────────────────────────────────────────────────────────────


class TestListModelsOpenRouter:
    def test_sorted_newest_first(self):
        resp = MagicMock()
        resp.json.return_value = {
            "data": [
                {"id": "a/model-old", "created": 100},
                {"id": "a/model-new", "created": 200},
            ]
        }
        resp.raise_for_status = MagicMock()
        with patch("openpurr.model_catalog.httpx.get", return_value=resp):
            names = model_catalog.list_models("openrouter")
        assert names == ["a/model-new", "a/model-old"]

    def test_returns_empty_on_exception(self):
        with patch(
            "openpurr.model_catalog.httpx.get", side_effect=httpx.ConnectError("x")
        ):
            assert model_catalog.list_models("openrouter") == []


# ─── unknown provider ───────────────────────────────────────────────────────────


def test_unknown_provider_returns_empty_list():
    assert model_catalog.list_models("imaginary_provider") == []
