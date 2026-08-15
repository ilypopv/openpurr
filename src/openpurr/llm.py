"""LLM provider factory."""

from __future__ import annotations

from openpurr.config import PROVIDER_BASE_URLS, Config
from openpurr.providers.base import BaseLLMProvider
from openpurr.providers.ollama import OllamaProvider

OPENAI_COMPATIBLE_PROVIDERS = frozenset(
    {"openai", "openrouter", "deepseek", "llamacpp", "mlx"}
)


def build_provider(config: Config) -> BaseLLMProvider:
    provider = config.llm_provider

    if provider == "ollama":
        return OllamaProvider(host=config.llm_host, model=config.llm_model)

    if provider in OPENAI_COMPATIBLE_PROVIDERS:
        from openpurr.providers.openai import OpenAICompatibleProvider

        base_url: str | None = (
            config.llm_host if config.llm_host != "http://localhost:11434" else None
        )
        base_url = base_url or PROVIDER_BASE_URLS.get(provider)
        return OpenAICompatibleProvider(
            api_key=config.llm_api_key,
            model=config.llm_model,
            base_url=base_url,
        )

    if provider == "anthropic":
        from openpurr.providers.anthropic import AnthropicProvider

        return AnthropicProvider(api_key=config.llm_api_key, model=config.llm_model)

    raise ValueError(
        f"Unsupported LLM provider: '{provider}'. "
        f"Supported: ollama, openai, anthropic, openrouter, deepseek, llamacpp, mlx"
    )
