"""Interactive first-run setup wizard for openpurr."""

from __future__ import annotations

import copy
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from openpurr.config import (
    CONFIG_PATH,
    DEFAULT_CONFIG,
    KNOWN_MODELS,
    PROVIDER_BASE_URLS,
    write_config,
)

console = Console()

PROVIDERS = [
    ("ollama", "Ollama (Free, runs locally)"),
    ("openai", "OpenAI (GPT-4o, GPT-4, …)"),
    ("anthropic", "Anthropic (Claude Opus, Sonnet, …)"),
    ("openrouter", "OpenRouter (Multiple providers)"),
    ("deepseek", "DeepSeek"),
    ("llamacpp", "llama.cpp (Free, runs locally)"),
    ("mlx", "MLX (Apple Silicon, local)"),
]

LOCAL_PROVIDERS = {"ollama", "llamacpp", "mlx"}


def _pick(options: list[tuple[str, str]], prompt: str, default: int = 1) -> str:
    console.print()
    for i, (_, label) in enumerate(options, 1):
        marker = "[bold cyan]>[/bold cyan]" if i == default else " "
        console.print(f"  {marker} [bold]{i}[/bold]. {label}")
    console.print()
    choices = [str(i) for i in range(1, len(options) + 1)]
    choice = Prompt.ask(prompt, choices=choices, default=str(default))
    return options[int(choice) - 1][0]


def _fetch_ollama_models(host: str) -> list[str]:
    try:
        import httpx

        r = httpx.get(f"{host.rstrip('/')}/api/tags", timeout=3.0)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


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

    # ── Select provider ──────────────────────────────────────────────────────
    console.print("[bold]Select your AI provider:[/bold]")
    provider = _pick(PROVIDERS, "Provider", default=1)

    data: dict[str, Any] = copy.deepcopy(DEFAULT_CONFIG)
    data["llm"]["provider"] = provider

    # ── Provider-specific configuration ──────────────────────────────────────
    if provider == "ollama":
        host = Prompt.ask(
            "Ollama host",
            default=data["llm"]["host"],
        )
        data["llm"]["host"] = host

        running = _fetch_ollama_models(host)
        if running:
            console.print(
                f"\n[dim]Detected {len(running)} local model(s): "
                + ", ".join(running[:5])
                + ("[dim]…[/dim]" if len(running) > 5 else "")
                + "[/dim]"
            )
            model_default = running[0]
        else:
            model_default = data["llm"]["model"]

        model = Prompt.ask("Model name", default=model_default)
        data["llm"]["model"] = model

        keep_alive = Prompt.ask(
            "Keep-alive duration (0s = unload VRAM immediately, 5m = keep warm)",
            default=data["llm"]["keep_alive"],
        )
        data["llm"]["keep_alive"] = keep_alive

    elif provider in ("llamacpp", "mlx"):
        default_host = PROVIDER_BASE_URLS.get(provider, "http://localhost:8080/v1")
        host = Prompt.ask("Server base URL", default=default_host)
        data["llm"]["host"] = host
        model = Prompt.ask("Model name (leave blank to use server default)", default="")
        data["llm"]["model"] = model
        data["llm"]["api_key"] = ""

    else:
        # Cloud providers that need an API key
        api_key = Prompt.ask("API key", password=True)
        if not api_key.strip():
            console.print("[red]API key is required. Setup aborted.[/red]")
            return False
        data["llm"]["api_key"] = api_key.strip()

        models = KNOWN_MODELS.get(provider, [])
        if models:
            console.print("\n[bold]Select a model:[/bold]")
            model_options = [(m, m) for m in models] + [
                ("__custom__", "Enter custom model name…")
            ]
            chosen = _pick(model_options, "Model", default=1)
            if chosen == "__custom__":
                chosen = Prompt.ask("Model name")
        else:
            chosen = Prompt.ask("Model name")
        data["llm"]["model"] = chosen

        # Custom base URL (optional)
        default_url = PROVIDER_BASE_URLS.get(provider, "")
        if default_url:
            data["llm"]["host"] = default_url
        custom_url = Prompt.ask(
            "Custom base URL (leave blank to use provider default)",
            default="",
        )
        if custom_url.strip():
            data["llm"]["host"] = custom_url.strip()

    # ── PR settings ───────────────────────────────────────────────────────────
    console.print()
    default_base = Prompt.ask("Default base branch", default=data["pr"]["default_base"])
    data["pr"]["default_base"] = default_base

    # ── Write config ──────────────────────────────────────────────────────────
    write_config(data)
    console.print(
        f"\n[bold green]Configuration saved to {CONFIG_PATH}[/bold green]\n"
        "Run [bold cyan]opo[/bold cyan] to generate your first PR description.\n"
        "Run [bold cyan]opo config describe[/bold cyan] to review all settings.\n"
    )
    return True
