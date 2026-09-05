"""PR text generation: init (title + description) and review (changes summary)."""

from __future__ import annotations

import re

from rich.console import Console

from openpurr.config import Config, language_name
from openpurr.llm import build_provider
from openpurr.utils import git

console = Console()

_THINK_BLOCK_RE = re.compile(
    r"<(?:think|thought|thinking|reasoning)[^>]*>.*?</(?:think|thought|thinking|reasoning)>",
    re.IGNORECASE | re.DOTALL,
)


def _strip_thinking(text: str) -> str:
    """Remove leaked thinking blocks from model output.

    Some reasoning models emit ``<think>...</think>`` (and variants like
    ``<thought>`` / ``<thinking>`` / ``<reasoning>``) inline even when
    instructed not to. This helper strips those blocks so only the final
    answer is shown.

    Args:
        text: Raw model output, possibly containing thinking tags.

    Returns:
        Text with all thinking blocks removed and stripped.
    """
    return _THINK_BLOCK_RE.sub("", text).strip()


def _init_system_prompt(language: str) -> str:
    """Build the system prompt for PR title and description generation.

    Args:
        language: ISO 639-1 language code for the output.

    Returns:
        System prompt string that instructs the LLM to follow the PR
        template in the requested language.
    """
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
    """Build the system prompt for review summary generation.

    Args:
        language: ISO 639-1 language code for the output.

    Returns:
        System prompt string that instructs the LLM to summarize changes
        since the last review in the requested language.
    """
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
    """Ensure a model is configured, exiting with a hint otherwise.

    Args:
        config: Resolved openpurr configuration.

    Raises:
        SystemExit: If ``config.llm_model`` is empty, after printing a hint.
    """
    if not config.llm_model:
        console.print(
            "[bold red]No model configured.[/bold red] Run [bold cyan]opo setup[/bold cyan] "
            "or [bold cyan]opo config set model <name>[/bold cyan]."
        )
        raise SystemExit(1)


def run_init(base: str | None, config: Config) -> None:
    """Generate and print a PR title and description from the git diff.

    Extracts the diff against ``base`` (or the configured default), checks
    for empty diffs, and delegates generation to the configured LLM provider.
    The result is printed to the console after stripping any thinking blocks.

    Args:
        base: Base branch to diff against. If ``None``, the value from
            ``config.pr_default_base`` is used.
        config: Resolved openpurr configuration.

    Raises:
        SystemExit: If no model is configured, no changes are detected, or a
            git error occurs.
    """
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
    """Generate and print a post-review changes summary.

    Extracts the diff for the last ``commits`` commits and delegates
    generation to the configured LLM provider.

    Args:
        commits: Number of recent commits to summarize (``HEAD~N..HEAD``).
        config: Resolved openpurr configuration.

    Raises:
        SystemExit: If no model is configured, no changes are detected, or a
            git error occurs.
    """
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
