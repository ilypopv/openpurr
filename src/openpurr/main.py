"""CLI entrypoint for openpurr (command: opo)."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from openpurr import model_catalog
from openpurr import pr as pr_module
from openpurr.config import (
    CONFIG_DESCRIPTIONS,
    SHORT_KEY_MAP,
    Config,
    get_config_value,
    is_first_run,
    load_config,
    set_config_value,
)

console = Console()

app = typer.Typer(
    name="opo",
    help="Generate PR title & description, or a post-review changes summary.",
    no_args_is_help=False,
    invoke_without_command=True,
)

config_app = typer.Typer(
    name="config",
    help="Get, set, or describe openpurr configuration.",
    invoke_without_command=True,
    no_args_is_help=True,
)
app.add_typer(config_app, name="config")


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _config() -> Config:
    return Config()


def _ensure_config() -> None:
    """Run setup wizard if ~/.openpurr does not exist yet."""
    if is_first_run():
        console.print(
            "[yellow]No configuration found — running setup wizard.[/yellow]\n"
        )
        from openpurr.setup_wizard import run_setup

        if not run_setup():
            raise SystemExit(1)


# ─── Default action: generate PR description ─────────────────────────────────


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    base: str | None = typer.Option(
        None, "--base", "-b", help="Base branch to diff against."
    ),
) -> None:
    """Generate PR title and description from the current diff."""
    if ctx.invoked_subcommand is not None:
        return
    _ensure_config()
    pr_module.run_init(base=base, config=_config())


# ─── opo review ──────────────────────────────────────────────────────────────


@app.command("review")
def review(
    commits: int = typer.Option(
        1, "--commits", "-c", help="Number of recent commits to summarize."
    ),
) -> None:
    """Generate a 'Changes since last review' summary from the last N commits."""
    _ensure_config()
    pr_module.run_review(commits=commits, config=_config())


# ─── opo setup ───────────────────────────────────────────────────────────────


@app.command("setup")
def setup() -> None:
    """Run the interactive setup wizard to (re)configure ~/.openpurr."""
    from openpurr.setup_wizard import run_setup

    if not run_setup():
        raise SystemExit(1)


# ─── opo models ──────────────────────────────────────────────────────────────


@app.command("models")
def models(
    provider: str | None = typer.Option(
        None, "--provider", "-p", help="Provider to list models for."
    ),
) -> None:
    """List available models for a provider."""
    cfg = _config()
    target = provider or cfg.llm_provider

    names = model_catalog.list_models(
        target, api_key=cfg.llm_api_key, host=cfg.llm_host
    )
    if not names:
        console.print(
            f"[yellow]No models found for '{target}'. "
            "Check connectivity/API key, or set one directly with:[/yellow] "
            "[bold cyan]opo config set model <name>[/bold cyan]"
        )
        return

    console.print(f"[bold]Models available for {target}:[/bold]")
    for name in names:
        marker = "[bold cyan]*[/bold cyan] " if name == cfg.llm_model else "  "
        console.print(f"{marker}{name}")


# ─── opo config describe ─────────────────────────────────────────────────────


@config_app.command("describe")
def config_describe() -> None:
    """Show all configuration keys with descriptions and current values."""
    data = load_config()
    table = Table(
        title="openpurr configuration", show_header=True, header_style="bold cyan"
    )
    table.add_column("Key", style="bold", no_wrap=True)
    table.add_column("Current Value", style="green")
    table.add_column("Description")

    for short_key, description in CONFIG_DESCRIPTIONS.items():
        env_key = SHORT_KEY_MAP[short_key]
        value = str(data.get(env_key, ""))
        if short_key == "api_key" and value:
            value = value[:4] + "…" + value[-4:] if len(value) > 8 else "****"
        table.add_row(short_key, value, description)

    console.print(table)


# ─── opo config get ───────────────────────────────────────────────────────────


@config_app.command("get")
def config_get(
    key: str = typer.Argument(..., help="Config key, e.g. model"),
) -> None:
    """Print the current value of a configuration key."""
    try:
        value = get_config_value(key)
        console.print(str(value))
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise SystemExit(1)


# ─── opo config set ───────────────────────────────────────────────────────────


@config_app.command("set")
def config_set(
    key: str = typer.Argument(..., help="Config key, e.g. model"),
    value: str = typer.Argument(..., help="New value to set"),
) -> None:
    """Update a configuration key in ~/.openpurr."""
    try:
        set_config_value(key, value)
        console.print(f"[green]Set[/green] [bold]{key}[/bold] = {value}")
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise SystemExit(1)


if __name__ == "__main__":
    app()
