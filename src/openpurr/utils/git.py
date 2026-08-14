"""Git diff extraction and noise filtering."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass

EXCLUDED_PATTERNS: tuple[str, ...] = (
    ".lock",
    "lock.json",
    "pnpm-lock.yaml",
    "package-lock.json",
    "yarn.lock",
    ".min.js",
    ".svg",
    ".map",
)


class GitError(RuntimeError):
    """Raised when a git command fails or the repository state is invalid."""


@dataclass
class GitDiffResult:
    diff: str
    changed_files: list[str]


def _run_git(args: list[str]) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout
    except FileNotFoundError as exc:
        raise GitError("git executable not found.") from exc
    except subprocess.CalledProcessError as exc:
        raise GitError(f"git command failed: {' '.join(args)}\n{exc.stderr}") from exc


def _is_excluded(path: str) -> bool:
    return any(p in path for p in EXCLUDED_PATTERNS)


def get_diff(base: str) -> GitDiffResult:
    output = _run_git(["diff", "--name-only", f"{base}...HEAD"])
    files = [
        f for f in (l.strip() for l in output.splitlines()) if f and not _is_excluded(f)
    ]
    if not files:
        return GitDiffResult(diff="", changed_files=[])
    diff = _run_git(["diff", f"{base}...HEAD", "--", *files])
    return GitDiffResult(diff=diff, changed_files=files)


def get_recent_commits_diff(commits: int) -> GitDiffResult:
    ref = f"HEAD~{commits}"
    output = _run_git(["diff", "--name-only", ref, "HEAD"])
    files = [
        f for f in (l.strip() for l in output.splitlines()) if f and not _is_excluded(f)
    ]
    if not files:
        return GitDiffResult(diff="", changed_files=[])
    diff = _run_git(["diff", ref, "HEAD", "--", *files])
    return GitDiffResult(diff=diff, changed_files=files)


def get_current_branch() -> str:
    return _run_git(["rev-parse", "--abbrev-ref", "HEAD"]).strip()
