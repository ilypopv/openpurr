"""Tests for openpurr.config — flat env-style read/write, get/set."""

from __future__ import annotations

import pytest

from openpurr.config import (
    DEFAULT_CONFIG,
    Config,
    _parse_env,
    _parse_prompts,
    _render_env,
    _split_config_text,
    get_config_value,
    is_first_run,
    load_config,
    load_prompts,
    resolve_base_url,
    set_config_value,
    write_config,
)

# ─── _parse_env ────────────────────────────────────────────────────────────────


class TestParseEnv:
    def test_basic_key_value(self):
        assert _parse_env("OPO_PROVIDER=ollama\n") == {"OPO_PROVIDER": "ollama"}

    def test_no_quoting(self):
        assert _parse_env("OPO_MODEL=gemma4:26b-mlx\n") == {
            "OPO_MODEL": "gemma4:26b-mlx"
        }

    def test_blank_lines_ignored(self):
        assert _parse_env("\nOPO_PROVIDER=ollama\n\n") == {"OPO_PROVIDER": "ollama"}

    def test_comments_ignored(self):
        assert _parse_env("# a comment\nOPO_PROVIDER=ollama\n") == {
            "OPO_PROVIDER": "ollama"
        }

    def test_lines_without_equals_ignored(self):
        assert _parse_env("not a kv line\nOPO_PROVIDER=ollama\n") == {
            "OPO_PROVIDER": "ollama"
        }

    def test_empty_value(self):
        assert _parse_env("OPO_API_KEY=\n") == {"OPO_API_KEY": ""}

    def test_whitespace_stripped(self):
        assert _parse_env("  OPO_PROVIDER = ollama  \n") == {"OPO_PROVIDER": "ollama"}


# ─── _render_env ───────────────────────────────────────────────────────────────


class TestRenderEnv:
    def test_no_quotes_no_sections(self):
        rendered = _render_env({"OPO_PROVIDER": "ollama", "OPO_MODEL": "gemma4:26b"})
        assert '"' not in rendered
        assert "[" not in rendered
        assert "OPO_PROVIDER=ollama" in rendered
        assert "OPO_MODEL=gemma4:26b" in rendered

    def test_round_trips_through_parse_env(self):
        data = {"OPO_PROVIDER": "openai", "OPO_TEMPERATURE": "0.5"}
        assert _parse_env(_render_env(data)) == data


# ─── is_first_run ──────────────────────────────────────────────────────────────


class TestIsFirstRun:
    def test_true_when_file_absent(self, monkeypatch, tmp_path):
        monkeypatch.setattr("openpurr.config.CONFIG_PATH", tmp_path / "no_such_file")
        assert is_first_run() is True

    def test_false_when_file_present(self, monkeypatch, tmp_path):
        p = tmp_path / ".openpurr"
        p.write_text("OPO_PROVIDER=ollama\n")
        monkeypatch.setattr("openpurr.config.CONFIG_PATH", p)
        assert is_first_run() is False


# ─── load_config ───────────────────────────────────────────────────────────────


class TestLoadConfig:
    def test_returns_defaults_when_no_file(self, monkeypatch, tmp_path):
        monkeypatch.setattr("openpurr.config.CONFIG_PATH", tmp_path / "missing")
        cfg = load_config()
        assert cfg["OPO_PROVIDER"] == DEFAULT_CONFIG["OPO_PROVIDER"]
        assert cfg["OPO_BASE"] == DEFAULT_CONFIG["OPO_BASE"]

    def test_file_values_override_defaults(self, monkeypatch, tmp_path):
        p = tmp_path / ".openpurr"
        p.write_text("OPO_MODEL=gpt-4o\nOPO_PROVIDER=openai\n")
        monkeypatch.setattr("openpurr.config.CONFIG_PATH", p)
        cfg = load_config()
        assert cfg["OPO_MODEL"] == "gpt-4o"
        assert cfg["OPO_PROVIDER"] == "openai"

    def test_unspecified_keys_keep_defaults(self, monkeypatch, tmp_path):
        p = tmp_path / ".openpurr"
        p.write_text("OPO_BASE=develop\n")
        monkeypatch.setattr("openpurr.config.CONFIG_PATH", p)
        cfg = load_config()
        assert cfg["OPO_BASE"] == "develop"
        assert cfg["OPO_PROVIDER"] == DEFAULT_CONFIG["OPO_PROVIDER"]


# ─── write_config / round-trip ─────────────────────────────────────────────────


class TestWriteConfig:
    def test_round_trip(self, monkeypatch, tmp_path):
        p = tmp_path / ".openpurr"
        monkeypatch.setattr("openpurr.config.CONFIG_PATH", p)
        data = {
            "OPO_PROVIDER": "anthropic",
            "OPO_MODEL": "claude-opus-4-8",
            "OPO_API_KEY": "sk-ant",
            "OPO_HOST": "",
            "OPO_TEMPERATURE": "0.0",
            "OPO_KEEP_ALIVE": "0s",
            "OPO_BASE": "develop",
        }
        write_config(data)
        loaded = load_config()
        assert loaded["OPO_PROVIDER"] == "anthropic"
        assert loaded["OPO_MODEL"] == "claude-opus-4-8"
        assert loaded["OPO_BASE"] == "develop"

    def test_file_has_no_quotes_or_sections(self, monkeypatch, tmp_path):
        p = tmp_path / ".openpurr"
        monkeypatch.setattr("openpurr.config.CONFIG_PATH", p)
        write_config(DEFAULT_CONFIG)
        content = p.read_text()
        assert '"' not in content
        assert "[" not in content


# ─── custom prompt overrides ────────────────────────────────────────────────────


class TestSplitConfigText:
    def test_no_delimiter_is_all_flat(self):
        flat, prompt = _split_config_text("OPO_PROVIDER=ollama\n")
        assert flat == "OPO_PROVIDER=ollama"
        assert prompt == ""

    def test_splits_on_delimiter_line(self):
        flat, prompt = _split_config_text(
            "OPO_PROVIDER=ollama\n---\nINIT PROMPT:\nHello\n"
        )
        assert flat == "OPO_PROVIDER=ollama"
        assert prompt == "INIT PROMPT:\nHello"


class TestParsePrompts:
    def test_parses_single_section(self):
        assert _parse_prompts("INIT PROMPT:\nLine one\nLine two") == {
            "init": "Line one\nLine two"
        }

    def test_parses_both_sections(self):
        prompts = _parse_prompts(
            "INIT PROMPT:\nCustom init\n\nREVIEW PROMPT:\nCustom review"
        )
        assert prompts == {"init": "Custom init", "review": "Custom review"}

    def test_header_matching_is_case_insensitive(self):
        assert _parse_prompts("init prompt:\nhi") == {"init": "hi"}

    def test_no_headers_returns_empty(self):
        assert _parse_prompts("just some text\nno headers here") == {}


class TestLoadPrompts:
    def test_returns_empty_when_no_file(self, monkeypatch, tmp_path):
        monkeypatch.setattr("openpurr.config.CONFIG_PATH", tmp_path / "missing")
        assert load_prompts() == {}

    def test_returns_empty_when_no_delimiter(self, monkeypatch, tmp_path):
        p = tmp_path / ".openpurr"
        p.write_text("OPO_PROVIDER=ollama\n")
        monkeypatch.setattr("openpurr.config.CONFIG_PATH", p)
        assert load_prompts() == {}

    def test_reads_custom_prompts(self, monkeypatch, tmp_path):
        p = tmp_path / ".openpurr"
        p.write_text(
            "OPO_PROVIDER=ollama\nOPO_MODEL=gemma4:26b\n"
            "\n---\nINIT PROMPT:\nBe extra terse.\n"
        )
        monkeypatch.setattr("openpurr.config.CONFIG_PATH", p)
        assert load_prompts() == {"init": "Be extra terse."}

    def test_flat_config_unaffected_by_prompt_section(self, monkeypatch, tmp_path):
        p = tmp_path / ".openpurr"
        p.write_text(
            "OPO_PROVIDER=ollama\n---\nINIT PROMPT:\nOPO_FAKE=should not parse\n"
        )
        monkeypatch.setattr("openpurr.config.CONFIG_PATH", p)
        assert "OPO_FAKE" not in load_config()


class TestWriteConfigPreservesPrompts:
    def test_write_config_keeps_existing_prompt_section(self, monkeypatch, tmp_path):
        p = tmp_path / ".openpurr"
        p.write_text("OPO_PROVIDER=ollama\n\n---\nINIT PROMPT:\nMy custom prompt.\n")
        monkeypatch.setattr("openpurr.config.CONFIG_PATH", p)
        data = dict(DEFAULT_CONFIG)
        data["OPO_PROVIDER"] = "openai"
        write_config(data)
        assert load_config()["OPO_PROVIDER"] == "openai"
        assert load_prompts() == {"init": "My custom prompt."}

    def test_write_config_writes_no_prompt_section_when_none_exists(
        self, monkeypatch, tmp_path
    ):
        p = tmp_path / ".openpurr"
        monkeypatch.setattr("openpurr.config.CONFIG_PATH", p)
        write_config(DEFAULT_CONFIG)
        assert "---" not in p.read_text()


# ─── set_config_value ──────────────────────────────────────────────────────────


class TestSetConfigValue:
    def test_sets_string(self, monkeypatch, tmp_path):
        p = tmp_path / ".openpurr"
        monkeypatch.setattr("openpurr.config.CONFIG_PATH", p)
        set_config_value("model", "gpt-4o-mini")
        assert load_config()["OPO_MODEL"] == "gpt-4o-mini"

    def test_stores_raw_string(self, monkeypatch, tmp_path):
        p = tmp_path / ".openpurr"
        monkeypatch.setattr("openpurr.config.CONFIG_PATH", p)
        set_config_value("temperature", "0.7")
        assert load_config()["OPO_TEMPERATURE"] == "0.7"

    def test_preserves_other_values_on_repeated_set(self, monkeypatch, tmp_path):
        p = tmp_path / ".openpurr"
        monkeypatch.setattr("openpurr.config.CONFIG_PATH", p)
        set_config_value("model", "first")
        set_config_value("provider", "openai")
        cfg = load_config()
        assert cfg["OPO_MODEL"] == "first"
        assert cfg["OPO_PROVIDER"] == "openai"

    def test_invalid_key_raises(self, monkeypatch, tmp_path):
        monkeypatch.setattr("openpurr.config.CONFIG_PATH", tmp_path / ".openpurr")
        with pytest.raises(ValueError, match="Unknown key"):
            set_config_value("badkey", "value")


# ─── get_config_value ──────────────────────────────────────────────────────────


class TestGetConfigValue:
    def test_returns_set_value(self, monkeypatch, tmp_path):
        p = tmp_path / ".openpurr"
        monkeypatch.setattr("openpurr.config.CONFIG_PATH", p)
        set_config_value("model", "my-model")
        assert get_config_value("model") == "my-model"

    def test_returns_default_for_existing_key(self, monkeypatch, tmp_path):
        monkeypatch.setattr("openpurr.config.CONFIG_PATH", tmp_path / "missing")
        assert get_config_value("base") == DEFAULT_CONFIG["OPO_BASE"]

    def test_raises_value_error_for_unknown_key(self, monkeypatch, tmp_path):
        monkeypatch.setattr("openpurr.config.CONFIG_PATH", tmp_path / "missing")
        with pytest.raises(ValueError):
            get_config_value("nonexistent_key_xyz")


# ─── resolve_base_url ──────────────────────────────────────────────────────────


class TestResolveBaseUrl:
    def test_custom_host_wins(self):
        assert (
            resolve_base_url("openai", "https://my-proxy.example.com/v1")
            == "https://my-proxy.example.com/v1"
        )

    def test_default_ollama_host_falls_back_to_provider_default(self):
        assert (
            resolve_base_url("openrouter", "http://localhost:11434")
            == "https://openrouter.ai/api/v1"
        )

    def test_no_host_falls_back_to_provider_default(self):
        assert resolve_base_url("deepseek", None) == "https://api.deepseek.com/v1"

    def test_gemini_falls_back_to_provider_default(self):
        assert (
            resolve_base_url("gemini", None)
            == "https://generativelanguage.googleapis.com/v1beta/openai/"
        )

    def test_unknown_provider_with_no_host_returns_none(self):
        assert resolve_base_url("openai", None) is None


# ─── Config class ──────────────────────────────────────────────────────────────


class TestConfigClass:
    def test_all_properties(self, monkeypatch, tmp_path):
        p = tmp_path / ".openpurr"
        p.write_text(
            "OPO_PROVIDER=openai\n"
            "OPO_MODEL=gpt-4o\n"
            "OPO_API_KEY=sk-x\n"
            "OPO_TEMPERATURE=0.2\n"
            "OPO_KEEP_ALIVE=0s\n"
            "OPO_BASE=develop\n"
        )
        monkeypatch.setattr("openpurr.config.CONFIG_PATH", p)
        cfg = Config()
        assert cfg.llm_provider == "openai"
        assert cfg.llm_model == "gpt-4o"
        assert cfg.llm_api_key == "sk-x"
        assert cfg.llm_temperature == pytest.approx(0.2)
        assert cfg.llm_keep_alive == "0s"
        assert cfg.pr_default_base == "develop"

    def test_host_falls_back_to_default(self, monkeypatch, tmp_path):
        monkeypatch.setattr("openpurr.config.CONFIG_PATH", tmp_path / "missing")
        cfg = Config()
        assert cfg.llm_host == "http://localhost:11434"

    def test_model_defaults_to_empty_string(self, monkeypatch, tmp_path):
        monkeypatch.setattr("openpurr.config.CONFIG_PATH", tmp_path / "missing")
        cfg = Config()
        assert cfg.llm_model == ""

    def test_custom_prompts_default_to_empty_string(self, monkeypatch, tmp_path):
        monkeypatch.setattr("openpurr.config.CONFIG_PATH", tmp_path / "missing")
        cfg = Config()
        assert cfg.custom_init_prompt == ""
        assert cfg.custom_review_prompt == ""

    def test_custom_prompts_read_from_file(self, monkeypatch, tmp_path):
        p = tmp_path / ".openpurr"
        p.write_text(
            "OPO_PROVIDER=ollama\n"
            "\n---\nINIT PROMPT:\nInit override.\n\nREVIEW PROMPT:\nReview override.\n"
        )
        monkeypatch.setattr("openpurr.config.CONFIG_PATH", p)
        cfg = Config()
        assert cfg.custom_init_prompt == "Init override."
        assert cfg.custom_review_prompt == "Review override."
