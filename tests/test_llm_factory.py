"""Tests for openpurr.llm.build_provider — routing, base URL, and API key wiring."""

from __future__ import annotations

import pytest

from openpurr.config import DEFAULT_CONFIG, PROVIDER_BASE_URLS, Config
from openpurr.llm import build_provider
from openpurr.providers.anthropic import AnthropicProvider
from openpurr.providers.ollama import OllamaProvider
from openpurr.providers.openai import OpenAICompatibleProvider

_SHORT_TO_ENV = {
    "llm.provider": "OPO_PROVIDER",
    "llm.model": "OPO_MODEL",
    "llm.api_key": "OPO_API_KEY",
    "llm.host": "OPO_HOST",
    "llm.temperature": "OPO_TEMPERATURE",
    "llm.keep_alive": "OPO_KEEP_ALIVE",
}


def _cfg(**overrides) -> Config:
    """Build a Config with specific overrides, e.g. _cfg(**{'llm.provider': 'openai'})."""
    data = dict(DEFAULT_CONFIG)
    for dotted, val in overrides.items():
        data[_SHORT_TO_ENV[dotted]] = val
    return Config(data)


# ─── provider class routing ───────────────────────────────────────────────────


class TestProviderRouting:
    def test_ollama_returns_ollama_provider(self):
        assert isinstance(
            build_provider(_cfg(**{"llm.provider": "ollama"})), OllamaProvider
        )

    def test_openai_returns_openai_compatible(self):
        p = build_provider(_cfg(**{"llm.provider": "openai", "llm.api_key": "sk-x"}))
        assert isinstance(p, OpenAICompatibleProvider)

    def test_anthropic_returns_anthropic_provider(self):
        p = build_provider(
            _cfg(**{"llm.provider": "anthropic", "llm.api_key": "ant-x"})
        )
        assert isinstance(p, AnthropicProvider)

    @pytest.mark.parametrize("provider", ["openrouter", "deepseek", "llamacpp", "mlx"])
    def test_all_openai_compatible_providers(self, provider):
        p = build_provider(_cfg(**{"llm.provider": provider, "llm.api_key": "k"}))
        assert isinstance(p, OpenAICompatibleProvider)

    def test_unknown_provider_raises_value_error(self):
        with pytest.raises(ValueError, match="Unsupported"):
            build_provider(_cfg(**{"llm.provider": "imaginary_provider"}))


# ─── base URL wiring ──────────────────────────────────────────────────────────


class TestBaseUrl:
    def test_ollama_uses_host_from_config(self):
        p = build_provider(
            _cfg(**{"llm.provider": "ollama", "llm.host": "http://custom:11434"})
        )
        assert p.host == "http://custom:11434"

    def test_openrouter_gets_provider_base_url(self):
        p = build_provider(_cfg(**{"llm.provider": "openrouter", "llm.api_key": "k"}))
        assert p._base_url == PROVIDER_BASE_URLS["openrouter"]

    def test_deepseek_gets_provider_base_url(self):
        p = build_provider(_cfg(**{"llm.provider": "deepseek", "llm.api_key": "k"}))
        assert p._base_url == PROVIDER_BASE_URLS["deepseek"]

    def test_llamacpp_gets_provider_base_url(self):
        p = build_provider(_cfg(**{"llm.provider": "llamacpp"}))
        assert p._base_url == PROVIDER_BASE_URLS["llamacpp"]

    def test_openai_has_no_base_url_by_default(self):
        # openai has no entry in PROVIDER_BASE_URLS; SDK uses its own default
        p = build_provider(_cfg(**{"llm.provider": "openai", "llm.api_key": "k"}))
        assert p._base_url is None

    def test_custom_host_overrides_provider_default(self):
        p = build_provider(
            _cfg(
                **{
                    "llm.provider": "openai",
                    "llm.api_key": "k",
                    "llm.host": "https://my-proxy.example.com/v1",
                }
            )
        )
        assert p._base_url == "https://my-proxy.example.com/v1"

    def test_custom_host_overrides_openrouter_default(self):
        p = build_provider(
            _cfg(
                **{
                    "llm.provider": "openrouter",
                    "llm.api_key": "k",
                    "llm.host": "https://custom-router.example.com/v1",
                }
            )
        )
        assert p._base_url == "https://custom-router.example.com/v1"


# ─── API key / model wiring ───────────────────────────────────────────────────


class TestCredentialWiring:
    def test_api_key_passed_to_openai_provider(self):
        p = build_provider(_cfg(**{"llm.provider": "openai", "llm.api_key": "sk-test"}))
        assert p._api_key == "sk-test"

    def test_api_key_passed_to_anthropic_provider(self):
        p = build_provider(
            _cfg(**{"llm.provider": "anthropic", "llm.api_key": "ant-test"})
        )
        assert p._api_key == "ant-test"

    def test_model_passed_to_ollama(self):
        p = build_provider(_cfg(**{"llm.provider": "ollama", "llm.model": "llama3:8b"}))
        assert p.model == "llama3:8b"

    def test_model_passed_to_openai_compatible(self):
        p = build_provider(
            _cfg(
                **{"llm.provider": "openai", "llm.api_key": "k", "llm.model": "gpt-4o"}
            )
        )
        assert p.model == "gpt-4o"

    def test_model_passed_to_anthropic(self):
        p = build_provider(
            _cfg(
                **{
                    "llm.provider": "anthropic",
                    "llm.api_key": "k",
                    "llm.model": "claude-opus-4-8",
                }
            )
        )
        assert p.model == "claude-opus-4-8"
