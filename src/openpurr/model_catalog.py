"""Live model discovery per provider.

No model names are hardcoded here — every provider exposes some form of a
models-list endpoint, and we query it directly so the tool never goes stale
when a provider ships a new model. Every branch swallows its own errors and
returns an empty list on failure; callers fall back to letting the user type
a model name manually.
"""

from __future__ import annotations

import httpx

from openpurr.config import PROVIDER_BASE_URLS, resolve_base_url

# Substrings that mark an OpenAI model id as non-chat (embeddings, audio,
# image, moderation, legacy completion families) — a heuristic filter, not a
# pinned model list.
_NON_CHAT_DENYLIST = (
    "embedding",
    "whisper",
    "tts",
    "dall-e",
    "moderation",
    "davinci",
    "babbage",
    "audio",
    "transcribe",
    "image",
)


def _list_ollama(host: str | None) -> list[str]:
    base = host or "http://localhost:11434"
    try:
        r = httpx.get(f"{base.rstrip('/')}/api/tags", timeout=5.0)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:  # noqa: BLE001 - never block callers on a flaky provider
        return []


def _list_openai_compatible(provider: str, api_key: str, host: str | None) -> list[str]:
    try:
        from openai import OpenAI

        base_url = resolve_base_url(provider, host)
        client = OpenAI(api_key=api_key or "not-needed", base_url=base_url)
        models = list(client.models.list())
    except Exception:  # noqa: BLE001 - never block callers on a flaky provider
        return []

    models.sort(key=lambda m: getattr(m, "created", 0) or 0, reverse=True)
    names = [m.id for m in models]
    if provider == "openai":
        names = [
            n for n in names if not any(bad in n.lower() for bad in _NON_CHAT_DENYLIST)
        ]
    return names


def _list_anthropic(api_key: str) -> list[str]:
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        models = list(client.models.list())
    except Exception:  # noqa: BLE001 - never block callers on a flaky provider
        return []

    models.sort(key=lambda m: getattr(m, "created_at", None) or "", reverse=True)
    return [m.id for m in models]


def _list_openrouter() -> list[str]:
    try:
        r = httpx.get(f"{PROVIDER_BASE_URLS['openrouter']}/models", timeout=5.0)
        r.raise_for_status()
        entries = r.json().get("data", [])
    except Exception:  # noqa: BLE001 - never block callers on a flaky provider
        return []

    entries.sort(key=lambda m: m.get("created", 0) or 0, reverse=True)
    return [m["id"] for m in entries if "id" in m]


def list_models(provider: str, api_key: str = "", host: str | None = None) -> list[str]:
    """Fetch the live list of model names for a provider. Never raises."""
    if provider == "ollama":
        return _list_ollama(host)
    if provider == "anthropic":
        return _list_anthropic(api_key)
    if provider == "openrouter":
        return _list_openrouter()
    if provider in ("openai", "deepseek", "llamacpp", "mlx"):
        return _list_openai_compatible(provider, api_key, host)
    return []
