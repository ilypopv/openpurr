"""PR text generation: init (title + description) and review (changes summary)."""

from __future__ import annotations

import re

from rich.console import Console

from openpurr.config import Config, language_name
from openpurr.llm import build_provider
from openpurr.utils import git

console = Console()

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)


def _strip_thinking(text: str) -> str:
    """Drop a leaked <think>...</think> reasoning block some models emit inline."""
    return _THINK_BLOCK_RE.sub("", text).strip()


def _init_system_prompt(language: str) -> str:
    return f"""\
You are a Senior Principal Engineer. Based on the provided git diff, generate a Pull Request Title and Description.

CRITICAL RULES:
1. Output ONLY the exact template below, filled in — nothing before, after, or between its parts. No preamble, reasoning, analysis, or commentary. Do not think out loud; if you reason internally, never print that reasoning. The Title is REQUIRED and is the first line of output — it is not preamble.
2. Language: Write entirely in {language_name(language)}.
3. Title MUST strictly follow Conventional Commits (e.g., feat(auth): add JWT refresh endpoint).
4. Description MUST strictly follow this Markdown structure.

TEMPLATE (replace the placeholders, keep everything else verbatim):

<title as plain text, Conventional Commits format>

## 📝 Summary
[2-3 sentences summarizing the core changes]

## 🛠 Type of Change
[Plain bullet list of every type that applies, chosen from: Bug fix, New feature, Refactoring, Breaking change. A PR can be more than one type — list all that fit. Do NOT use checkboxes or brackets like "- [ ]", just "- Bug fix".]

## 🔍 Key Changes
- [Bullet points of technical details]

## 🧪 How Has This Been Tested?
- [Testing details derived from diff or code]
"""


def _review_system_prompt(language: str) -> str:
    return f"""\
You are a Senior Principal Engineer. Based on the git diff of changes made during code review, write a concise summary for the reviewer.

CRITICAL RULES:
1. Output ONLY the content below — no preamble, reasoning, analysis, or commentary of any kind. Do not think out loud; if you reason internally, never print that reasoning.
2. Language: Write entirely in {language_name(language)}.
3. Follow this structure strictly:

## 🔄 Changes since last review
- [Bullet points describing fixes and requested updates]
"""


# Default (English) renderings — kept as module-level constants so callers that
# want the built-in prompt without a Config instance (e.g. tests) still have a
# stable name to import; actual generation always resolves via config.llm_language.
INIT_SYSTEM_PROMPT = _init_system_prompt("en")
REVIEW_SYSTEM_PROMPT = _review_system_prompt("en")


def _require_model(config: Config) -> None:
    if not config.llm_model:
        console.print(
            "[bold red]No model configured.[/bold red] Run [bold cyan]opo setup[/bold cyan] "
            "or [bold cyan]opo config set model <name>[/bold cyan]."
        )
        raise SystemExit(1)


def run_init(base: str | None, config: Config) -> None:
    _require_model(config)
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

    with console.status(
        "[bold green]Generating PR title and description...[/bold green]"
    ):
        output = provider.generate(
            prompt=diff_result.diff,
            system_prompt=config.custom_init_prompt
            or _init_system_prompt(config.llm_language),
            temperature=config.llm_temperature,
            keep_alive=config.llm_keep_alive,
        )

    console.print()
    console.print(_strip_thinking(output))


def run_review(commits: int, config: Config) -> None:
    _require_model(config)
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

    with console.status("[bold green]Generating review summary...[/bold green]"):
        output = provider.generate(
            prompt=diff_result.diff,
            system_prompt=config.custom_review_prompt
            or _review_system_prompt(config.llm_language),
            temperature=config.llm_temperature,
            keep_alive=config.llm_keep_alive,
        )

    console.print()
    console.print(_strip_thinking(output))
