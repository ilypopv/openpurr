"""Interactive first-run setup wizard for openpurr."""

from __future__ import annotations

import questionary
from rich.console import Console
from rich.panel import Panel

from openpurr import model_catalog
from openpurr.config import (
    CONFIG_PATH,
    DEFAULT_CONFIG,
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
    """Raised when the user cancels a prompt (Ctrl-C / Esc)."""


def _select(message: str, choices, default=None) -> str:
    result = questionary.select(
        message, choices=choices, default=default, style=_STYLE
    ).ask()
    if result is None:
        raise _SetupAborted
    return result


def _text(message: str, default: str = "") -> str:
    result = questionary.text(message, default=default, style=_STYLE).ask()
    if result is None:
        raise _SetupAborted
    return result


def _password(message: str) -> str:
    result = questionary.password(message, style=_STYLE).ask()
    if result is None:
        raise _SetupAborted
    return result


def _pick_model(models: list[str], default_text: str = "") -> str:
    if not models:
        return _text("Model name", default=default_text)
    chosen = _select("Select a model:", choices=[*models, CUSTOM_MODEL_CHOICE])
    if chosen == CUSTOM_MODEL_CHOICE:
        return _text("Model name", default=default_text)
    return chosen


def _validate_model(provider: str, model: str, partial: dict[str, str]) -> str | None:
    """Try one real, minimal request against `model`. None on success, else the error text.

    No provider's model-list endpoint reliably marks a model as still usable —
    OpenAI/Anthropic expose no deprecation flag, and Gemini's "no longer
    available to new users" restriction only surfaces at request time. Actually
    calling the model is the only trustworthy check.
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
    return data


def run_setup() -> bool:
    """
    Run the interactive setup wizard.
    Writes ~/.openpurr and returns True on success, False if the user aborts.
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
