"""Interactive first-run setup wizard for openpurr."""

from __future__ import annotations

import questionary
from rich.console import Console
from rich.panel import Panel

from openpurr import model_catalog
from openpurr.config import (
    CONFIG_PATH,
    DEFAULT_CONFIG,
    LANGUAGES,
    PROVIDER_BASE_URLS,
    Config,
    write_config,
)
from openpurr.llm import build_provider

console = Console()

PROVIDERS = [
    ("ollama", "Ollama (Free, runs locally)"),
    ("openai", "OpenAI (GPT models)"),
    ("anthropic", "Anthropic (Claude models)"),
    ("gemini", "Google (Gemini models)"),
    ("openrouter", "OpenRouter (Multiple providers)"),
    ("deepseek", "DeepSeek"),
    ("llamacpp", "llama.cpp (Free, runs locally)"),
    ("mlx", "MLX (Apple Silicon, local)"),
]

CUSTOM_MODEL_CHOICE = "Enter custom model name…"
CUSTOM_LANGUAGE_CHOICE = "Enter custom language code…"

# The focused row is marked by the "»" pointer alone — no reverse-video/
# background highlight on the row text itself. prompt_toolkit's own
# default_ui_style() defines `class:selected` as "reverse" (its generic
# text-selection class, reused here by questionary for the row matching
# `default=`); an empty style string doesn't clear an inherited attribute
# (empty means "unset", not "off"), so this must say "noreverse" explicitly.
_STYLE = questionary.Style(
    [
        ("qmark", "fg:cyan bold"),
        ("question", "bold"),
        ("answer", "fg:cyan bold"),
        ("pointer", "fg:cyan bold"),
        ("highlighted", "noreverse"),
        ("selected", "noreverse"),
        ("instruction", ""),
        ("text", ""),
    ]
)


class _SetupAborted(Exception):
    """Raised when the user cancels a prompt via Ctrl-C or Esc."""


def _select(message: str, choices, default=None) -> str:
    """Prompt the user to select from a list of choices.

    Args:
        message: Prompt message to display.
        choices: List of ``questionary.Choice`` or string options.
        default: Default choice to pre-select.

    Returns:
        The selected value.

    Raises:
        _SetupAborted: If the user cancels the prompt.
    """
    result = questionary.select(
        message, choices=choices, default=default, style=_STYLE
    ).ask()
    if result is None:
        raise _SetupAborted
    return result


def _text(message: str, default: str = "") -> str:
    """Prompt the user for free-text input.

    Args:
        message: Prompt message to display.
        default: Default value to show in the input.

    Returns:
        The entered text.

    Raises:
        _SetupAborted: If the user cancels the prompt.
    """
    result = questionary.text(message, default=default, style=_STYLE).ask()
    if result is None:
        raise _SetupAborted
    return result


def _password(message: str) -> str:
    """Prompt the user for a password / API key.

    Args:
        message: Prompt message to display.

    Returns:
        The entered password string.

    Raises:
        _SetupAborted: If the user cancels the prompt.
    """
    result = questionary.password(message, style=_STYLE).ask()
    if result is None:
        raise _SetupAborted
    return result


def _pick_model(models: list[str], default_text: str = "") -> str:
    """Let the user pick a model from a list or enter a custom name.

    Args:
        models: List of available model names. Empty list prompts for a
            custom name directly.
        default_text: Default text for the custom input prompt.

    Returns:
        Selected or entered model name.

    Raises:
        _SetupAborted: If the user cancels the prompt.
    """
    if not models:
        return _text("Model name", default=default_text)
    chosen = _select("Select a model:", choices=[*models, CUSTOM_MODEL_CHOICE])
    if chosen == CUSTOM_MODEL_CHOICE:
        return _text("Model name", default=default_text)
    return chosen


def _pick_language(default: str = "en") -> str:
    """Let the user pick an output language.

    Args:
        default: Default language code to pre-select.

    Returns:
        Selected language code.

    Raises:
        _SetupAborted: If the user cancels the prompt.
    """
    choices = [
        questionary.Choice(title=f"{name} ({code})", value=code)
        for code, name in LANGUAGES.items()
    ]
    chosen = _select(
        "Output language for generated PR text/reviews:",
        choices=[*choices, CUSTOM_LANGUAGE_CHOICE],
        default=next((c for c in choices if c.value == default), choices[0]),
    )
    if chosen == CUSTOM_LANGUAGE_CHOICE:
        return _text("Language code (ISO 639-1, e.g. en, ru, es, fr)", default=default)
    return chosen


def _validate_model(provider: str, model: str, partial: dict[str, str]) -> str | None:
    """Try a minimal request to verify that a model is usable.

    No provider's model-list endpoint reliably marks a model as still usable —
    OpenAI/Anthropic expose no deprecation flag, and Gemini's "no longer
    available to new users" restriction only surfaces at request time. Actually
    calling the model is the only trustworthy check.

    Args:
        provider: Provider key.
        model: Model name to test.
        partial: Partial config dict to build a trial provider (must contain
            at least the provider-specific auth/host keys).

    Returns:
        ``None`` on success, otherwise the error message as a string.
    """
    trial = build_provider(
        Config({**partial, "OPO_PROVIDER": provider, "OPO_MODEL": model})
    )
    try:
        trial.generate(prompt="ping", system_prompt="Reply with: pong", temperature=0.0)
    except Exception as exc:  # noqa: BLE001 - any provider failure means the model isn't usable
        return str(exc)
    return None


def _pick_and_validate_model(
    provider: str, models: list[str], partial: dict[str, str], default_text: str = ""
) -> str:
    """Prompt for a model with live validation and retry/skip logic.

    Args:
        provider: Provider key.
        models: List of available model names.
        partial: Partial config dict for trial validation.
        default_text: Default text for custom model input.

    Returns:
        Selected model name, validated or explicitly skipped.

    Raises:
        _SetupAborted: If the user cancels the prompt.
    """
    remaining = list(models)
    while True:
        model = _pick_model(remaining, default_text=default_text)
        console.print(f"[dim]Verifying {model} works…[/dim]")
        error = _validate_model(provider, model, partial)
        if error is None:
            return model
        console.print(f"[red]{model} isn't usable: {error}[/red]")
        choice = _select(
            "What would you like to do?",
            choices=[
                questionary.Choice("Pick a different model", value="retry"),
                questionary.Choice("Save anyway (skip verification)", value="skip"),
            ],
        )
        if choice == "skip":
            return model
        remaining = [m for m in remaining if m != model]


def _run_setup_flow() -> dict[str, str]:
    """Run the interactive provider and preference selection flow.

    Returns:
        Flat config dictionary ready to be written via :func:`write_config`.

    Raises:
        _SetupAborted: If the user cancels any prompt.
    """
    data = dict(DEFAULT_CONFIG)

    console.print("[bold]Select your AI provider:[/bold]")
    provider_choices = [
        questionary.Choice(title=label, value=key) for key, label in PROVIDERS
    ]
    provider = _select(
        "Provider", choices=provider_choices, default=provider_choices[0]
    )
    data["OPO_PROVIDER"] = provider

    if provider == "ollama":
        host = _text("Ollama host", default=data["OPO_HOST"])
        data["OPO_HOST"] = host

        console.print("[dim]Checking for local models…[/dim]")
        models = model_catalog.list_models("ollama", host=host)
        model = _pick_model(models)
        data["OPO_MODEL"] = model

        data["OPO_KEEP_ALIVE"] = _text(
            "Keep-alive duration (0s = unload VRAM immediately, 5m = keep warm)",
            default=data["OPO_KEEP_ALIVE"],
        )

    elif provider in ("llamacpp", "mlx"):
        default_host = PROVIDER_BASE_URLS.get(provider, "http://localhost:8080/v1")
        host = _text("Server base URL", default=default_host)
        data["OPO_HOST"] = host
        data["OPO_API_KEY"] = ""

        console.print("[dim]Checking for available models…[/dim]")
        models = model_catalog.list_models(provider, host=host)
        data["OPO_MODEL"] = _pick_model(models)

    else:
        api_key = _password("API key")
        if not api_key.strip():
            console.print("[red]API key is required. Setup aborted.[/red]")
            raise _SetupAborted
        data["OPO_API_KEY"] = api_key.strip()

        default_url = PROVIDER_BASE_URLS.get(provider, "")
        if default_url:
            data["OPO_HOST"] = default_url

        console.print("[dim]Fetching available models…[/dim]")
        models = model_catalog.list_models(
            provider, api_key=data["OPO_API_KEY"], host=data.get("OPO_HOST")
        )
        data["OPO_MODEL"] = _pick_and_validate_model(provider, models, data)

        custom_url = _text(
            "Custom base URL (leave blank to use provider default)", default=""
        )
        if custom_url.strip():
            data["OPO_HOST"] = custom_url.strip()

    data["OPO_BASE"] = _text("Default base branch", default=data["OPO_BASE"])
    data["OPO_LANGUAGE"] = _pick_language(default=data["OPO_LANGUAGE"])
    return data


def run_setup() -> bool:
    """Run the interactive setup wizard.

    Writes the configuration to :data:`CONFIG_PATH` via the wizard flow.

    Returns:
        True if setup completed and the config was written, False if the
        user aborted.
    """
    console.print(
        Panel(
            "[bold]Welcome to openpurr![/bold]\n"
            "Let's configure your AI provider. "
            "This will create [cyan]~/.openpurr[/cyan].",
            title="[bold cyan]openpurr setup[/bold cyan]",
            expand=False,
        )
    )

    try:
        data = _run_setup_flow()
    except _SetupAborted:
        console.print("[red]Setup cancelled.[/red]")
        return False

    write_config(data)
    console.print(
        f"\n[bold green]Configuration saved to {CONFIG_PATH}[/bold green]\n"
        "Run [bold cyan]opo[/bold cyan] to generate your first PR description.\n"
        "Run [bold cyan]opo config describe[/bold cyan] to review all settings.\n"
    )
    return True
