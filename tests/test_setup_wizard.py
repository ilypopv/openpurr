"""Tests for openpurr.setup_wizard — flow logic, with the prompt layer mocked.

`_select`/`_text`/`_password` wrap questionary (which needs a real TTY), so
these tests drive `run_setup()` end-to-end by feeding canned answers through
those three functions and asserting on the config that gets written.
"""

from __future__ import annotations

from unittest.mock import patch

from openpurr import setup_wizard


def _run_with_answers(select_answers, text_answers, password_answers=None, models=None):
    select_iter = iter(select_answers)
    text_iter = iter(text_answers)
    password_iter = iter(password_answers or [])

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
            select_answers=["ollama", "llama3:8b"],
            text_answers=["http://localhost:11434", "5m", "main"],
            models=["llama3:8b", "gemma:2b"],
        )
        assert ok is True
        assert written["OPO_PROVIDER"] == "ollama"
        assert written["OPO_MODEL"] == "llama3:8b"
        assert written["OPO_HOST"] == "http://localhost:11434"
        assert written["OPO_KEEP_ALIVE"] == "5m"
        assert written["OPO_BASE"] == "main"

    def test_falls_back_to_text_entry_when_no_models_found(self):
        ok, written = _run_with_answers(
            select_answers=["ollama"],
            text_answers=["http://localhost:11434", "custom-model", "5m", "main"],
            models=[],
        )
        assert ok is True
        assert written["OPO_MODEL"] == "custom-model"

    def test_custom_model_sentinel_falls_through_to_text(self):
        ok, written = _run_with_answers(
            select_answers=["ollama", setup_wizard.CUSTOM_MODEL_CHOICE],
            text_answers=["http://localhost:11434", "hand-typed-model", "5m", "main"],
            models=["llama3:8b"],
        )
        assert ok is True
        assert written["OPO_MODEL"] == "hand-typed-model"


class TestCloudProviderFlow:
    def test_openai_flow_requires_api_key_and_fetches_model(self):
        ok, written = _run_with_answers(
            select_answers=["openai", "gpt-5"],
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
            select_answers=["gemini", "gemini-2.0-flash"],
            text_answers=["", "main"],
            password_answers=["gm-test"],
            models=["gemini-2.0-flash", "gemini-2.0-pro"],
        )
        assert ok is True
        assert written["OPO_PROVIDER"] == "gemini"
        assert written["OPO_API_KEY"] == "gm-test"
        assert written["OPO_MODEL"] == "gemini-2.0-flash"

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
            select_answers=["anthropic", "claude-x"],
            text_answers=["https://my-proxy.example.com", "main"],
            password_answers=["sk-ant"],
            models=["claude-x"],
        )
        assert ok is True
        assert written["OPO_HOST"] == "https://my-proxy.example.com"


class TestLocalServerFlow:
    def test_llamacpp_flow_has_no_api_key(self):
        ok, written = _run_with_answers(
            select_answers=["llamacpp", "local-model"],
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
