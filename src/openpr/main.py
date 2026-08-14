"""CLI entrypoint for openpr (command: opo)."""

from __future__ import annotations

import typer
from rich.console import Console

from openpr import pr as pr_module
from openpr.config import Config

console = Console()

app = typer.Typer(
    name="opo",
    help="Generate PR title & description, or a post-review changes summary.",
    no_args_is_help=False,
    invoke_without_command=True,
)


def _config() -> Config:
    return Config()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    base: str | None = typer.Option(
        None, "--base", "-b", help="Base branch to diff against."
    ),
    unload: bool = typer.Option(
        False, "--unload", help="Flush model from VRAM after generation."
    ),
) -> None:
    """Generate PR title and description from the current diff."""
    if ctx.invoked_subcommand is not None:
        return
    pr_module.run_init(base=base, unload=unload, config=_config())


@app.command("review")
def review(
    commits: int = typer.Option(
        1, "--commits", "-c", help="Number of recent commits to summarize."
    ),
    unload: bool = typer.Option(
        False, "--unload", help="Flush model from VRAM after generation."
    ),
) -> None:
    """Generate a 'Changes since last review' summary from the last N commits."""
    pr_module.run_review(commits=commits, unload=unload, config=_config())


if __name__ == "__main__":
    app()
