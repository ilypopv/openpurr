"""Tests for openpurr.utils.git — exclusion filter and diff extraction."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from openpurr.utils.git import (
    GitError,
    _is_excluded,
    get_diff,
    get_recent_commits_diff,
)

# ─── _is_excluded ─────────────────────────────────────────────────────────────


class TestIsExcluded:
    @pytest.mark.parametrize(
        "path",
        [
            "package-lock.json",
            "yarn.lock",
            "Pipfile.lock",
            "pnpm-lock.yaml",
            "some/nested/package-lock.json",
            "dist/app.min.js",
            "assets/logo.svg",
            "dist/bundle.js.map",
            "lock.json",
        ],
    )
    def test_excluded_files(self, path):
        assert _is_excluded(path) is True

    @pytest.mark.parametrize(
        "path",
        [
            "src/main.py",
            "README.md",
            "pyproject.toml",
            "src/components/Button.tsx",
            "tests/test_config.py",
            "docs/index.html",
            "Makefile",
        ],
    )
    def test_normal_files_allowed(self, path):
        assert _is_excluded(path) is False


# ─── get_diff ─────────────────────────────────────────────────────────────────


class TestGetDiff:
    def _fake_run(self, name_only_output: str, diff_output: str = "diff content"):
        """Returns a side_effect function that routes by args."""

        def _run(args, **kwargs):
            m = MagicMock()
            m.stdout = name_only_output if "--name-only" in args else diff_output
            return m

        return _run

    def test_empty_when_no_changed_files(self):
        with patch("openpurr.utils.git.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="")
            result = get_diff("main")
        assert result.diff == ""
        assert result.changed_files == []

    def test_lockfiles_excluded_from_changed_files(self):
        with patch(
            "openpurr.utils.git.subprocess.run",
            side_effect=self._fake_run("src/main.py\npackage-lock.json\nyarn.lock\n"),
        ):
            result = get_diff("main")
        assert "src/main.py" in result.changed_files
        assert "package-lock.json" not in result.changed_files
        assert "yarn.lock" not in result.changed_files

    def test_svg_files_excluded(self):
        with patch(
            "openpurr.utils.git.subprocess.run",
            side_effect=self._fake_run("src/app.py\nassets/logo.svg\n"),
        ):
            result = get_diff("main")
        assert result.changed_files == ["src/app.py"]

    def test_all_excluded_returns_empty(self):
        with patch(
            "openpurr.utils.git.subprocess.run",
            side_effect=self._fake_run("package-lock.json\nyarn.lock\n"),
        ):
            result = get_diff("main")
        assert result.diff == ""
        assert result.changed_files == []

    def test_diff_content_returned(self):
        with patch(
            "openpurr.utils.git.subprocess.run",
            side_effect=self._fake_run("src/app.py\n", diff_output="--- a\n+++ b\n"),
        ):
            result = get_diff("main")
        assert result.diff == "--- a\n+++ b\n"

    def test_base_ref_used_in_git_args(self):
        calls = []

        def fake_run(args, **kwargs):
            calls.append(list(args))
            m = MagicMock()
            m.stdout = "src/app.py\n" if "--name-only" in args else "diff"
            return m

        with patch("openpurr.utils.git.subprocess.run", side_effect=fake_run):
            get_diff("develop")

        all_args = " ".join(str(a) for c in calls for a in c)
        assert "develop...HEAD" in all_args

    def test_raises_git_error_on_process_failure(self):
        with (
            patch(
                "openpurr.utils.git.subprocess.run",
                side_effect=subprocess.CalledProcessError(
                    128, "git", stderr="fatal error"
                ),
            ),
            pytest.raises(GitError),
        ):
            get_diff("main")

    def test_raises_git_error_when_git_not_found(self):
        with (
            patch("openpurr.utils.git.subprocess.run", side_effect=FileNotFoundError()),
            pytest.raises(GitError, match="not found"),
        ):
            get_diff("main")


# ─── get_recent_commits_diff ──────────────────────────────────────────────────


class TestGetRecentCommitsDiff:
    def test_uses_correct_head_ref(self):
        calls = []

        def fake_run(args, **kwargs):
            calls.append(list(args))
            m = MagicMock()
            m.stdout = "src/foo.py\n" if "--name-only" in args else "diff"
            return m

        with patch("openpurr.utils.git.subprocess.run", side_effect=fake_run):
            get_recent_commits_diff(3)

        all_args = " ".join(str(a) for c in calls for a in c)
        assert "HEAD~3" in all_args

    def test_uses_head_tilde_one_for_single_commit(self):
        calls = []

        def fake_run(args, **kwargs):
            calls.append(list(args))
            m = MagicMock()
            m.stdout = "src/foo.py\n" if "--name-only" in args else "diff"
            return m

        with patch("openpurr.utils.git.subprocess.run", side_effect=fake_run):
            get_recent_commits_diff(1)

        all_args = " ".join(str(a) for c in calls for a in c)
        assert "HEAD~1" in all_args

    def test_empty_when_no_changes(self):
        with patch("openpurr.utils.git.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="")
            result = get_recent_commits_diff(1)
        assert result.diff == ""
        assert result.changed_files == []

    def test_lockfiles_still_excluded(self):
        def fake_run(args, **kwargs):
            m = MagicMock()
            m.stdout = (
                "src/app.py\npnpm-lock.yaml\n" if "--name-only" in args else "diff"
            )
            return m

        with patch("openpurr.utils.git.subprocess.run", side_effect=fake_run):
            result = get_recent_commits_diff(2)

        assert "src/app.py" in result.changed_files
        assert "pnpm-lock.yaml" not in result.changed_files
