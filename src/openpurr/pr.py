"""PR text generation: init (title + description) and review (changes summary)."""

from __future__ import annotations

from openpurr.config import Config
from openpurr.llm import build_provider
from openpurr.utils import git
from rich.console import Console

console = Console()

INIT_SYSTEM_PROMPT = """\
You are a Senior Principal Engineer. Based on the provided git diff, generate a Pull Request Title and Description.

CRITICAL RULES:
1. Language: English ONLY.
2. Title MUST strictly follow Conventional Commits (e.g., feat(auth): add JWT refresh endpoint).
3. Description MUST strictly follow this Markdown structure:

## 📝 Summary
[2-3 sentences summarizing the core changes]

## 🛠 Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Refactoring
- [ ] Breaking change

## 🔍 Key Changes
- [Bullet points of technical details]

## 🧪 How Has This Been Tested?
- [Testing details derived from diff or code]
"""

REVIEW_SYSTEM_PROMPT = """\
You are a Senior Principal Engineer. Based on the git diff of changes made during code review, write a concise summary for the reviewer.

CRITICAL RULES:
1. Language: English ONLY.
2. Follow this structure strictly:

## 🔄 Changes since last review
- [Bullet points describing fixes and requested updates]
"""


def run_init(base: str | None, unload: bool, config: Config) -> None:
    base_ref = base or config.pr_default_base
    console.print(f"[bold cyan]Extracting diff against '{base_ref}'...[/bold cyan]")

    try:
        diff_result = git.get_diff(base_ref)
    except git.GitError as exc:
        console.print(f"[bold red]Git error:[/bold red] {exc}")
        raise SystemExit(1) from exc

    if not diff_result.diff.strip():
        console.print("[yellow]No changes detected against the base branch.[/yellow]")
        raise SystemExit(0)

    console.print(f"[dim]{len(diff_result.changed_files)} file(s) changed.[/dim]")

    provider = build_provider(config)
    keep_alive = "0" if unload else config.llm_keep_alive

    with console.status(
        "[bold green]Generating PR title and description...[/bold green]"
    ):
        output = provider.generate(
            prompt=diff_result.diff,
            system_prompt=INIT_SYSTEM_PROMPT,
            temperature=config.llm_temperature,
            keep_alive=keep_alive,
        )

    console.print()
    console.print(output.strip())


def run_review(commits: int, unload: bool, config: Config) -> None:
    console.print(
        f"[bold cyan]Extracting diff for the last {commits} commit(s)...[/bold cyan]"
    )

    try:
        diff_result = git.get_recent_commits_diff(commits)
    except git.GitError as exc:
        console.print(f"[bold red]Git error:[/bold red] {exc}")
        raise SystemExit(1) from exc

    if not diff_result.diff.strip():
        console.print(
            "[yellow]No changes detected in the specified commit range.[/yellow]"
        )
        raise SystemExit(0)

    console.print(f"[dim]{len(diff_result.changed_files)} file(s) changed.[/dim]")

    provider = build_provider(config)
    keep_alive = "0" if unload else config.llm_keep_alive

    with console.status("[bold green]Generating review summary...[/bold green]"):
        output = provider.generate(
            prompt=diff_result.diff,
            system_prompt=REVIEW_SYSTEM_PROMPT,
            temperature=config.llm_temperature,
            keep_alive=keep_alive,
        )

    console.print()
    console.print(output.strip())
