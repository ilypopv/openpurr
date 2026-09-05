"""LLM provider factory.

Selects and instantiates the appropriate :class:`BaseLLMProvider`
implementation based on :class:`openpurr.config.Config`.
"""

from __future__ import annotations

from openpurr.config import Config, resolve_base_url
from openpurr.providers.base import BaseLLMProvider
from openpurr.providers.ollama import OllamaProvider

OPENAI_COMPATIBLE_PROVIDERS = frozenset(
    {"openai", "gemini", "openrouter", "deepseek", "llamacpp", "mlx"}
)


def build_provider(config: Config) -> BaseLLMProvider:
    """Build an LLM provider from resolved config.

    Args:
        config: Resolved openpurr configuration.

    Returns:
        An instance of :class:`BaseLLMProvider` for the configured provider.

    Raises:
        ValueError: If the provider key is not supported.
    """
    provider = config.llm_provider

    if provider == "ollama":
        return OllamaProvider(host=config.llm_host, model=config.llm_model)

    if provider in OPENAI_COMPATIBLE_PROVIDERS:
        from openpurr.providers.openai import OpenAICompatibleProvider

        base_url = resolve_base_url(provider, config.llm_host)
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
        f"Supported: ollama, openai, anthropic, gemini, openrouter, deepseek, llamacpp, mlx"
    )
