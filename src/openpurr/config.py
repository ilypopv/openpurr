"""Configuration loader and writer for openpurr (~/.openpurr).

Storage format is flat ``OPO_KEY=value`` lines (no sections, no quoting) —
mirrors opencommit's ``~/.opencommit`` rather than TOML. Everything after a
line containing only ``---`` is a free-text area for custom system prompt
overrides (see :data:`PROMPT_HEADERS`), kept out of the ``OPO_KEY=value``
parsing entirely so prompt text can contain ``=``, brackets, anything.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

CONFIG_PATH = Path.home() / ".openpurr"

DEFAULT_CONFIG: dict[str, str] = {
    "OPO_PROVIDER": "ollama",
    "OPO_MODEL": "",
    "OPO_API_KEY": "",
    "OPO_HOST": "http://localhost:11434",
    "OPO_TEMPERATURE": "0.0",
    "OPO_KEEP_ALIVE": "5m",
    "OPO_BASE": "main",
    "OPO_LANGUAGE": "en",
}

# Short CLI keys → OPO_* env keys
SHORT_KEY_MAP: dict[str, str] = {
    "provider": "OPO_PROVIDER",
    "model": "OPO_MODEL",
    "api_key": "OPO_API_KEY",
    "host": "OPO_HOST",
    "temperature": "OPO_TEMPERATURE",
    "keep_alive": "OPO_KEEP_ALIVE",
    "base": "OPO_BASE",
    "language": "OPO_LANGUAGE",
}

CONFIG_DESCRIPTIONS: dict[str, str] = {
    "provider": "LLM provider — ollama | openai | anthropic | gemini | openrouter | deepseek | llamacpp | mlx",
    "model": "Model name (fetched live per-provider via `opo models`)",
    "api_key": "API key for cloud providers (not required for ollama / llamacpp / mlx)",
    "host": "Base URL — Ollama default is http://localhost:11434; custom endpoint for other providers",
    "temperature": "Sampling temperature: 0.0 = deterministic",
    "keep_alive": "Ollama VRAM keep-alive duration — 0s = unload immediately, 5m = keep 5 min",
    "base": "Default base branch to diff against (main, master, …)",
    "language": "Output language for generated PR text (ISO 639-1 code, e.g. en, ru, es, fr, de, ja, zh)",
}

# ISO 639-1 code → English name, used to phrase the language instruction in the
# default system prompts. Not exhaustive — unknown codes pass through as-is,
# so any code the LLM understands works even if it isn't listed here.
LANGUAGES: dict[str, str] = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "uk": "Ukrainian",
    "pl": "Polish",
    "nl": "Dutch",
    "tr": "Turkish",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
    "ar": "Arabic",
    "hi": "Hindi",
    "vi": "Vietnamese",
    "id": "Indonesian",
    "cs": "Czech",
    "sv": "Swedish",
}


def language_name(code: str) -> str:
    """Return a human-readable language name for a prompt instruction.

    Unknown codes are returned unchanged so any code the LLM understands
    works even if not listed in :data:`LANGUAGES`.

    Args:
        code: ISO 639-1 code, case-insensitive, possibly with whitespace.

    Returns:
        English language name if known, otherwise the original code as-is.
    """
    return LANGUAGES.get(code.strip().lower(), code)


PROMPT_DELIMITER = "---"

# Free-text section headers (case-insensitive) → the Config property key they fill.
PROMPT_HEADERS: dict[str, str] = {
    "INIT PROMPT:": "init",
    "REVIEW PROMPT:": "review",
}

PROVIDER_BASE_URLS: dict[str, str] = {
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "openrouter": "https://openrouter.ai/api/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "llamacpp": "http://localhost:8080/v1",
    "mlx": "http://localhost:8080/v1",
}


def resolve_base_url(provider: str, host: str | None) -> str | None:
    """Resolve the effective base URL for a provider.

    An explicit non-default ``host`` wins over provider defaults. The default
    ``OPO_HOST`` (``http://localhost:11434``) does not count as custom and
    falls back to the provider's own default.

    Args:
        provider: Provider key (e.g. ``"gemini"``, ``"openai"``).
        host: Configured host value, or ``None`` if unset.

    Returns:
        Effective base URL string, or ``None`` if no override applies
        (used for providers that rely on SDK defaults like OpenAI).
    """
    if host and host != DEFAULT_CONFIG["OPO_HOST"]:
        return host
    return PROVIDER_BASE_URLS.get(provider)


def is_first_run() -> bool:
    """Check whether openpurr has been configured before.

    Returns:
        True if the config file at :data:`CONFIG_PATH` does not exist,
        False otherwise.
    """
    return not CONFIG_PATH.exists()


def _parse_env(text: str) -> dict[str, str]:
    """Parse flat ``KEY=value`` lines.

    Blank lines, comments (``#``), and lines without ``=`` are ignored.
    Whitespace around keys and values is stripped. No quoting or escaping
    is performed.

    Args:
        text: Raw config file content (flat part only).

    Returns:
        Dictionary of parsed key-value pairs.
    """
    data: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        data[key.strip()] = value.strip()
    return data


def _render_env(data: dict[str, str]) -> str:
    """Render a config dictionary to flat ``KEY=value`` text.

    Args:
        data: Mapping of ``OPO_*`` keys to values.

    Returns:
        Newline-terminated string with one ``KEY=value`` per line.
    """
    return "\n".join(f"{key}={value}" for key, value in data.items()) + "\n"


def _split_config_text(text: str) -> tuple[str, str]:
    """Split raw file text at the ``---`` delimiter.

    Args:
        text: Full content of the config file.

    Returns:
        Tuple ``(flat_part, prompt_part)``. If no delimiter is present,
        ``prompt_part`` is an empty string.
    """
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == PROMPT_DELIMITER:
            return "\n".join(lines[:i]), "\n".join(lines[i + 1 :])
    return "\n".join(lines), ""


def _parse_prompts(text: str) -> dict[str, str]:
    """Parse custom prompt overrides from the free-text section.

    Recognizes ``INIT PROMPT:`` and ``REVIEW PROMPT:`` headers
    case-insensitively via :data:`PROMPT_HEADERS`.

    Args:
        text: Prompt section text (content after ``---``).

    Returns:
        Mapping with optional keys ``"init"`` and ``"review"`` to their
        stripped prompt bodies.
    """
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        key = PROMPT_HEADERS.get(line.strip().upper())
        if key is not None:
            current = key
            sections[current] = []
            continue
        if current is not None:
            sections[current].append(line)
    return {key: "\n".join(lines).strip() for key, lines in sections.items()}


def _render_prompts(prompts: dict[str, str]) -> str:
    """Render prompt overrides to the delimited free-text format.

    Args:
        prompts: Mapping with optional ``"init"`` and ``"review"`` keys.

    Returns:
        String starting with ``---`` followed by present prompt sections.
    """
    headers = {v: k for k, v in PROMPT_HEADERS.items()}
    parts = [PROMPT_DELIMITER]
    for key in ("init", "review"):
        if prompts.get(key):
            parts.extend([headers[key], prompts[key], ""])
    return "\n".join(parts).rstrip() + "\n"


def write_config(data: dict[str, str]) -> None:
    """Write config data to disk, preserving existing prompt overrides.

    Reads current prompts via :func:`load_prompts` and re-appends them after
    the flat ``KEY=value`` section.

    Args:
        data: Mapping of ``OPO_*`` keys to values to persist.
    """
    text = _render_env(data)
    prompts = load_prompts()
    if prompts:
        text = text.rstrip("\n") + "\n\n" + _render_prompts(prompts)
    CONFIG_PATH.write_text(text)


def load_config() -> dict[str, str]:
    """Load config from disk with defaults.

    Args:
        None.

    Returns:
        Dictionary merging :data:`DEFAULT_CONFIG` with any values found in
        the flat part of the config file. Returns defaults unchanged if the
        file does not exist.
    """
    if not CONFIG_PATH.exists():
        return dict(DEFAULT_CONFIG)
    flat_text, _ = _split_config_text(CONFIG_PATH.read_text())
    return {**DEFAULT_CONFIG, **_parse_env(flat_text)}


def load_prompts() -> dict[str, str]:
    """Load custom system prompt overrides from disk.

    Returns:
        Mapping with optional ``"init"`` and ``"review"`` keys. Empty dict
        if the config file does not exist or has no delimited section.
    """
    if not CONFIG_PATH.exists():
        return {}
    _, prompt_text = _split_config_text(CONFIG_PATH.read_text())
    return _parse_prompts(prompt_text)


def _resolve_key(key: str) -> str:
    """Resolve a short CLI key to its ``OPO_*`` env form.

    Args:
        key: Short key (e.g. ``"model"``) or already-qualified ``"OPO_MODEL"``.

    Returns:
        Qualified ``OPO_*`` key. Unknown short keys are returned unchanged
        and will be rejected by the caller.
    """
    return SHORT_KEY_MAP.get(key, key)


def set_config_value(key: str, value: str) -> None:
    """Set a single config value and persist to disk.

    Args:
        key: Short key (``"provider"``, ``"model"``, etc.) or ``"OPO_*"``.
        value: Value to store as a string.

    Raises:
        ValueError: If ``key`` is not a known config key.
    """
    env_key = _resolve_key(key)
    if env_key not in DEFAULT_CONFIG:
        raise ValueError(f"Unknown key {key!r}. Valid keys: {', '.join(SHORT_KEY_MAP)}")
    data = load_config()
    data[env_key] = value
    write_config(data)


def get_config_value(key: str) -> Any:
    """Get a single config value.

    Args:
        key: Short key (``"provider"``, ``"model"``, etc.) or ``"OPO_*"``.

    Returns:
        Stored value for the key, or the default if not set.

    Raises:
        ValueError: If ``key`` is not a known config key.
    """
    env_key = _resolve_key(key)
    if env_key not in DEFAULT_CONFIG:
        raise ValueError(f"Unknown key {key!r}. Valid keys: {', '.join(SHORT_KEY_MAP)}")
    return load_config()[env_key]


class Config:
    """Resolved openpurr configuration.

    Wraps the flat ``OPO_*`` dict and prompt overrides with typed accessors.
    When instantiated without arguments, loads from disk; when ``data`` is
    provided explicitly, it never touches the filesystem unless ``prompts``
    are also passed.

    Attributes:
        llm_provider: Provider key (``"ollama"``, ``"openai"``, etc.).
        llm_model: Model identifier.
        llm_api_key: API key for cloud providers.
        llm_host: Base URL / host string.
        llm_temperature: Sampling temperature as float.
        llm_keep_alive: Ollama VRAM keep-alive string.
        pr_default_base: Default base branch for PR diffs.
        llm_language: ISO 639-1 language code.
        custom_init_prompt: Override for the init system prompt.
        custom_review_prompt: Override for the review system prompt.
    """

    def __init__(
        self,
        data: dict[str, str] | None = None,
        prompts: dict[str, str] | None = None,
    ) -> None:
        """Initialize configuration.

        Args:
            data: Optional flat config dict. If ``None``, loads from disk via
                :func:`load_config`.
            prompts: Optional prompt overrides. If ``None`` and ``data`` is
                ``None``, loads from disk via :func:`load_prompts`;
                otherwise defaults to empty dict to avoid filesystem access.
        """
        self._data = data if data is not None else load_config()
        if prompts is not None:
            self._prompts = prompts
        else:
            # Only auto-load from disk alongside auto-loaded data — an explicit
            # `data` dict (tests, one-off trial configs) must never touch the
            # filesystem unless prompts are explicitly passed too.
            self._prompts = load_prompts() if data is None else {}

    @property
    def llm_provider(self) -> str:
        """LLM provider key."""
        return self._data.get("OPO_PROVIDER", DEFAULT_CONFIG["OPO_PROVIDER"])

    @property
    def llm_model(self) -> str:
        """Model identifier."""
        return self._data.get("OPO_MODEL", DEFAULT_CONFIG["OPO_MODEL"])

    @property
    def llm_api_key(self) -> str:
        """API key for cloud providers."""
        return self._data.get("OPO_API_KEY", "")

    @property
    def llm_host(self) -> str:
        """Base URL / host for the provider."""
        return self._data.get("OPO_HOST", DEFAULT_CONFIG["OPO_HOST"])

    @property
    def llm_temperature(self) -> float:
        """Sampling temperature as float."""
        return float(
            self._data.get("OPO_TEMPERATURE", DEFAULT_CONFIG["OPO_TEMPERATURE"])
        )

    @property
    def llm_keep_alive(self) -> str:
        """Ollama VRAM keep-alive duration string."""
        return self._data.get("OPO_KEEP_ALIVE", DEFAULT_CONFIG["OPO_KEEP_ALIVE"])

    @property
    def pr_default_base(self) -> str:
        """Default base branch to diff against."""
        return self._data.get("OPO_BASE", DEFAULT_CONFIG["OPO_BASE"])

    @property
    def llm_language(self) -> str:
        """Output language code (ISO 639-1)."""
        return self._data.get("OPO_LANGUAGE", DEFAULT_CONFIG["OPO_LANGUAGE"])

    @property
    def custom_init_prompt(self) -> str:
        """Custom system prompt for ``opo`` (init)."""
        return self._prompts.get("init", "")

    @property
    def custom_review_prompt(self) -> str:
        """Custom system prompt for ``opo review``."""
        return self._prompts.get("review", "")
