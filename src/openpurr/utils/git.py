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


def _parse_name_status(output: str) -> tuple[list[str], list[str]]:
    """Parse `git diff --name-status` output.

    Returns (changed_files, diff_paths) where:
    - changed_files is the de-duplicated display list (new name for renames).
    - diff_paths is every path that must be passed to `git diff -- <paths>`
      to preserve rename detection (both old+new for R/C entries).
    """
    changed_files: list[str] = []
    diff_paths: list[str] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        # Real --name-status uses tab separation: "M\tpath" or "R100\told\tnew".
        # Fall back to treating a bare path as "M\tpath" for test compat.
        if "\t" not in line:
            paths = [line.strip()]
        else:
            parts = line.split("\t")
            # parts[0] is status like "M", "A", "R100", "C75"; remaining are paths
            paths = parts[1:] if len(parts) > 1 else []
        filtered = [p for p in paths if p and not _is_excluded(p)]
        if not filtered:
            continue
        diff_paths.extend(filtered)
        # For renames/copies the last path is the new name — that's what
        # users expect in the "1 file(s) changed" summary. For normal
        # entries filtered has exactly one element.
        changed_files.append(filtered[-1])
    return changed_files, diff_paths


def get_diff(base: str) -> GitDiffResult:
    output = _run_git(["diff", "--name-status", f"{base}...HEAD"])
    changed_files, diff_paths = _parse_name_status(output)
    if not changed_files:
        return GitDiffResult(diff="", changed_files=[])
    diff = _run_git(["diff", f"{base}...HEAD", "--", *diff_paths])
    return GitDiffResult(diff=diff, changed_files=changed_files)


def get_recent_commits_diff(commits: int) -> GitDiffResult:
    ref = f"HEAD~{commits}"
    output = _run_git(["diff", "--name-status", ref, "HEAD"])
    changed_files, diff_paths = _parse_name_status(output)
    if not changed_files:
        return GitDiffResult(diff="", changed_files=[])
    diff = _run_git(["diff", ref, "HEAD", "--", *diff_paths])
    return GitDiffResult(diff=diff, changed_files=changed_files)


def get_current_branch() -> str:
    return _run_git(["rev-parse", "--abbrev-ref", "HEAD"]).strip()
