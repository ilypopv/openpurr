"""Tests for openpurr.setup_wizard — flow logic, with the prompt layer mocked.

`_select`/`_text`/`_password` wrap questionary (which needs a real TTY), so
these tests drive `run_setup()` end-to-end by feeding canned answers through
those three functions and asserting on the config that gets written.
"""

from __future__ import annotations

from itertools import repeat
from unittest.mock import patch

from openpurr import setup_wizard


def _run_with_answers(
    select_answers,
    text_answers,
    password_answers=None,
    models=None,
    validation_errors=None,
):
    select_iter = iter(select_answers)
    text_iter = iter(text_answers)
    password_iter = iter(password_answers or [])
    validation_iter = (
        iter(validation_errors) if validation_errors is not None else repeat(None)
    )

    written: dict = {}

    def fake_write_config(data):
        written.update(data)

    with (
        patch.object(
            setup_wizard, "_select", side_effect=lambda *a, **k: next(select_iter)
        ),
        patch.object(
            setup_wizard, "_text", side_effect=lambda *a, **k: next(text_iter)
        ),
        patch.object(
            setup_wizard, "_password", side_effect=lambda *a, **k: next(password_iter)
        ),
        patch.object(setup_wizard, "write_config", side_effect=fake_write_config),
        patch.object(
            setup_wizard,
            "_validate_model",
            side_effect=lambda *a, **k: next(validation_iter),
        ),
        patch(
            "openpurr.model_catalog.list_models",
            return_value=models if models is not None else [],
        ),
    ):
        result = setup_wizard.run_setup()

    return result, written


class TestOllamaFlow:
    def test_writes_expected_config_with_fetched_model(self):
        ok, written = _run_with_answers(
            select_answers=["ollama", "llama3:8b", "en"],
            text_answers=["http://localhost:11434", "5m", "main"],
            models=["llama3:8b", "gemma:2b"],
        )
        assert ok is True
        assert written["OPO_PROVIDER"] == "ollama"
        assert written["OPO_MODEL"] == "llama3:8b"
        assert written["OPO_HOST"] == "http://localhost:11434"
        assert written["OPO_KEEP_ALIVE"] == "5m"
        assert written["OPO_BASE"] == "main"
        assert written["OPO_LANGUAGE"] == "en"

    def test_falls_back_to_text_entry_when_no_models_found(self):
        ok, written = _run_with_answers(
            select_answers=["ollama", "en"],
            text_answers=["http://localhost:11434", "custom-model", "5m", "main"],
            models=[],
        )
        assert ok is True
        assert written["OPO_MODEL"] == "custom-model"

    def test_custom_model_sentinel_falls_through_to_text(self):
        ok, written = _run_with_answers(
            select_answers=["ollama", setup_wizard.CUSTOM_MODEL_CHOICE, "en"],
            text_answers=["http://localhost:11434", "hand-typed-model", "5m", "main"],
            models=["llama3:8b"],
        )
        assert ok is True
        assert written["OPO_MODEL"] == "hand-typed-model"


class TestCloudProviderFlow:
    def test_openai_flow_requires_api_key_and_fetches_model(self):
        ok, written = _run_with_answers(
            select_answers=["openai", "gpt-5", "en"],
            text_answers=["", "main"],
            password_answers=["sk-test"],
            models=["gpt-5", "gpt-5-mini"],
        )
        assert ok is True
        assert written["OPO_PROVIDER"] == "openai"
        assert written["OPO_API_KEY"] == "sk-test"
        assert written["OPO_MODEL"] == "gpt-5"

    def test_gemini_flow_requires_api_key_and_fetches_model(self):
        ok, written = _run_with_answers(
            select_answers=["gemini", "gemini-2.0-flash", "en"],
            text_answers=["", "main"],
            password_answers=["gm-test"],
            models=["gemini-2.0-flash", "gemini-2.0-pro"],
        )
        assert ok is True
        assert written["OPO_PROVIDER"] == "gemini"
        assert written["OPO_API_KEY"] == "gm-test"
        assert written["OPO_MODEL"] == "gemini-2.0-flash"

    def test_invalid_model_prompts_retry_and_picks_next(self):
        ok, written = _run_with_answers(
            select_answers=[
                "gemini",
                "gemini-2.5-flash",
                "retry",
                "gemini-3.6-flash",
                "en",
            ],
            text_answers=["", "main"],
            password_answers=["gm-test"],
            models=["gemini-2.5-flash", "gemini-3.6-flash"],
            validation_errors=["model not found", None],
        )
        assert ok is True
        assert written["OPO_MODEL"] == "gemini-3.6-flash"

    def test_invalid_model_can_be_saved_anyway(self):
        ok, written = _run_with_answers(
            select_answers=["gemini", "gemini-2.5-flash", "skip", "en"],
            text_answers=["", "main"],
            password_answers=["gm-test"],
            models=["gemini-2.5-flash"],
            validation_errors=["model not found"],
        )
        assert ok is True
        assert written["OPO_MODEL"] == "gemini-2.5-flash"

    def test_blank_api_key_aborts_setup(self):
        ok, written = _run_with_answers(
            select_answers=["openai"],
            text_answers=[],
            password_answers=["   "],
        )
        assert ok is False
        assert written == {}

    def test_custom_base_url_overrides_provider_default(self):
        ok, written = _run_with_answers(
            select_answers=["anthropic", "claude-x", "en"],
            text_answers=["https://my-proxy.example.com", "main"],
            password_answers=["sk-ant"],
            models=["claude-x"],
        )
        assert ok is True
        assert written["OPO_HOST"] == "https://my-proxy.example.com"


class TestLanguageStep:
    def test_defaults_to_english(self):
        ok, written = _run_with_answers(
            select_answers=["ollama", "llama3:8b", "en"],
            text_answers=["http://localhost:11434", "5m", "main"],
            models=["llama3:8b"],
        )
        assert ok is True
        assert written["OPO_LANGUAGE"] == "en"

    def test_picks_a_listed_language(self):
        ok, written = _run_with_answers(
            select_answers=["ollama", "llama3:8b", "es"],
            text_answers=["http://localhost:11434", "5m", "main"],
            models=["llama3:8b"],
        )
        assert ok is True
        assert written["OPO_LANGUAGE"] == "es"

    def test_custom_language_sentinel_falls_through_to_text(self):
        ok, written = _run_with_answers(
            select_answers=[
                "ollama",
                "llama3:8b",
                setup_wizard.CUSTOM_LANGUAGE_CHOICE,
            ],
            text_answers=["http://localhost:11434", "5m", "main", "pt-br"],
            models=["llama3:8b"],
        )
        assert ok is True
        assert written["OPO_LANGUAGE"] == "pt-br"


class TestValidateModel:
    def test_returns_none_on_successful_call(self):
        with patch.object(setup_wizard, "build_provider") as mock_build:
            mock_build.return_value.generate.return_value = "pong"
            error = setup_wizard._validate_model(
                "gemini", "gemini-3.6-flash", {"OPO_API_KEY": "k"}
            )
        assert error is None

    def test_returns_error_text_on_failure(self):
        with patch.object(setup_wizard, "build_provider") as mock_build:
            mock_build.return_value.generate.side_effect = RuntimeError(
                "model not found"
            )
            error = setup_wizard._validate_model(
                "gemini", "gemini-2.5-flash", {"OPO_API_KEY": "k"}
            )
        assert error == "model not found"


class TestLocalServerFlow:
    def test_llamacpp_flow_has_no_api_key(self):
        ok, written = _run_with_answers(
            select_answers=["llamacpp", "local-model", "en"],
            text_answers=["http://localhost:8080/v1", "main"],
            models=["local-model"],
        )
        assert ok is True
        assert written["OPO_API_KEY"] == ""
        assert written["OPO_MODEL"] == "local-model"


class TestAbort:
    def test_cancelled_select_returns_false_without_writing(self):
        with (
            patch.object(
                setup_wizard, "_select", side_effect=setup_wizard._SetupAborted
            ),
            patch.object(setup_wizard, "write_config") as mock_write,
        ):
            assert setup_wizard.run_setup() is False
        mock_write.assert_not_called()
