"""Tests for openpurr.pr — that custom prompt overrides reach provider.generate().

`build_provider` and `git.get_diff`/`get_recent_commits_diff` are mocked; the
point of these tests is only which system_prompt string ends up passed to
generate(), not diff extraction or provider routing (covered elsewhere).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from openpurr import pr
from openpurr.config import DEFAULT_CONFIG, Config
from openpurr.utils.git import GitDiffResult


def _cfg(language: str = "en", **prompt_overrides) -> Config:
    data = {**DEFAULT_CONFIG, "OPO_MODEL": "test-model", "OPO_LANGUAGE": language}
    return Config(data, prompts=prompt_overrides)


class TestRunInit:
    def test_uses_default_prompt_when_no_override(self):
        fake_provider = MagicMock()
        fake_provider.generate.return_value = "output"
        with (
            patch.object(
                pr.git,
                "get_diff",
                return_value=GitDiffResult(diff="diff", changed_files=["a.py"]),
            ),
            patch.object(pr, "build_provider", return_value=fake_provider),
        ):
            pr.run_init(base="main", config=_cfg())
        _, kwargs = fake_provider.generate.call_args
        assert kwargs["system_prompt"] == pr.INIT_SYSTEM_PROMPT

    def test_uses_custom_init_prompt_when_set(self):
        fake_provider = MagicMock()
        fake_provider.generate.return_value = "output"
        with (
            patch.object(
                pr.git,
                "get_diff",
                return_value=GitDiffResult(diff="diff", changed_files=["a.py"]),
            ),
            patch.object(pr, "build_provider", return_value=fake_provider),
        ):
            pr.run_init(base="main", config=_cfg(init="Custom init prompt"))
        _, kwargs = fake_provider.generate.call_args
        assert kwargs["system_prompt"] == "Custom init prompt"

    def test_uses_configured_language_in_default_prompt(self):
        fake_provider = MagicMock()
        fake_provider.generate.return_value = "output"
        with (
            patch.object(
                pr.git,
                "get_diff",
                return_value=GitDiffResult(diff="diff", changed_files=["a.py"]),
            ),
            patch.object(pr, "build_provider", return_value=fake_provider),
        ):
            pr.run_init(base="main", config=_cfg(language="es"))
        _, kwargs = fake_provider.generate.call_args
        assert "Write entirely in Spanish." in kwargs["system_prompt"]
        assert kwargs["system_prompt"] != pr.INIT_SYSTEM_PROMPT

    def test_custom_init_prompt_overrides_language_setting(self):
        fake_provider = MagicMock()
        fake_provider.generate.return_value = "output"
        with (
            patch.object(
                pr.git,
                "get_diff",
                return_value=GitDiffResult(diff="diff", changed_files=["a.py"]),
            ),
            patch.object(pr, "build_provider", return_value=fake_provider),
        ):
            pr.run_init(
                base="main", config=_cfg(language="es", init="Custom init prompt")
            )
        _, kwargs = fake_provider.generate.call_args
        assert kwargs["system_prompt"] == "Custom init prompt"


class TestRunReview:
    def test_uses_default_prompt_when_no_override(self):
        fake_provider = MagicMock()
        fake_provider.generate.return_value = "output"
        with (
            patch.object(
                pr.git,
                "get_recent_commits_diff",
                return_value=GitDiffResult(diff="diff", changed_files=["a.py"]),
            ),
            patch.object(pr, "build_provider", return_value=fake_provider),
        ):
            pr.run_review(commits=1, config=_cfg())
        _, kwargs = fake_provider.generate.call_args
        assert kwargs["system_prompt"] == pr.REVIEW_SYSTEM_PROMPT

    def test_uses_custom_review_prompt_when_set(self):
        fake_provider = MagicMock()
        fake_provider.generate.return_value = "output"
        with (
            patch.object(
                pr.git,
                "get_recent_commits_diff",
                return_value=GitDiffResult(diff="diff", changed_files=["a.py"]),
            ),
            patch.object(pr, "build_provider", return_value=fake_provider),
        ):
            pr.run_review(commits=1, config=_cfg(review="Custom review prompt"))
        _, kwargs = fake_provider.generate.call_args
        assert kwargs["system_prompt"] == "Custom review prompt"

    def test_uses_configured_language_in_default_prompt(self):
        fake_provider = MagicMock()
        fake_provider.generate.return_value = "output"
        with (
            patch.object(
                pr.git,
                "get_recent_commits_diff",
                return_value=GitDiffResult(diff="diff", changed_files=["a.py"]),
            ),
            patch.object(pr, "build_provider", return_value=fake_provider),
        ):
            pr.run_review(commits=1, config=_cfg(language="fr"))
        _, kwargs = fake_provider.generate.call_args
        assert "Write entirely in French." in kwargs["system_prompt"]
        assert kwargs["system_prompt"] != pr.REVIEW_SYSTEM_PROMPT
