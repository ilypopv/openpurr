"""Tests for openpurr.config — TOML rendering, merge, read/write, get/set."""

from __future__ import annotations

import tomllib

import pytest

from openpurr.config import (
    DEFAULT_CONFIG,
    Config,
    _deep_merge,
    _render_toml,
    get_config_value,
    is_first_run,
    load_config,
    set_config_value,
    write_config,
)

# ─── _deep_merge ─────────────────────────────────────────────────────────────


class TestDeepMerge:
    def test_override_wins(self):
        base = {"llm": {"model": "base-model", "provider": "ollama"}}
        result = _deep_merge(base, {"llm": {"model": "override-model"}})
        assert result["llm"]["model"] == "override-model"
        assert result["llm"]["provider"] == "ollama"

    def test_nested_dict_merged_not_replaced(self):
        base = {"llm": {"a": 1, "b": 2}}
        result = _deep_merge(base, {"llm": {"b": 99}})
        assert result["llm"]["a"] == 1
        assert result["llm"]["b"] == 99

    def test_new_top_level_section_added(self):
        base = {"pr": {"default_base": "main"}}
        result = _deep_merge(base, {"extra": {"x": 1}})
        assert "extra" in result
        assert result["pr"]["default_base"] == "main"

    def test_scalar_override_replaces(self):
        base = {"llm": {"provider": "ollama"}}
        result = _deep_merge(base, {"llm": {"provider": "openai"}})
        assert result["llm"]["provider"] == "openai"

    def test_base_unchanged(self):
        base = {"llm": {"model": "original"}}
        _deep_merge(base, {"llm": {"model": "changed"}})
        assert base["llm"]["model"] == "original"


# ─── _render_toml ─────────────────────────────────────────────────────────────


class TestRenderToml:
    def test_string_values_quoted(self):
        toml = _render_toml({"llm": {"provider": "ollama", "model": "gemma4:26b"}})
        assert "[llm]" in toml
        assert 'provider = "ollama"' in toml
        assert 'model = "gemma4:26b"' in toml

    def test_float_rendered(self):
        assert "temperature = 0.0" in _render_toml({"llm": {"temperature": 0.0}})

    def test_bool_lowercase(self):
        toml = _render_toml({"flags": {"enabled": True, "disabled": False}})
        assert "enabled = true" in toml
        assert "disabled = false" in toml

    def test_round_trips_through_tomllib(self):
        data = {
            "llm": {"provider": "openai", "temperature": 0.5, "keep_alive": "5m"},
            "pr": {"default_base": "develop"},
        }
        parsed = tomllib.loads(_render_toml(data))
        assert parsed["llm"]["provider"] == "openai"
        assert parsed["llm"]["temperature"] == pytest.approx(0.5)
        assert parsed["pr"]["default_base"] == "develop"

    def test_special_chars_escaped(self):
        data = {"llm": {"api_key": 'sk-"test"\\path'}}
        parsed = tomllib.loads(_render_toml(data))
        assert parsed["llm"]["api_key"] == 'sk-"test"\\path'

    def test_each_section_has_header(self):
        toml = _render_toml({"llm": {"a": "1"}, "pr": {"b": "2"}})
        assert "[llm]" in toml
        assert "[pr]" in toml


# ─── is_first_run ─────────────────────────────────────────────────────────────


class TestIsFirstRun:
    def test_true_when_file_absent(self, monkeypatch, tmp_path):
        monkeypatch.setattr("openpurr.config.CONFIG_PATH", tmp_path / "no_such_file")
        assert is_first_run() is True

    def test_false_when_file_present(self, monkeypatch, tmp_path):
        p = tmp_path / ".openpurr"
        p.write_text('[llm]\nprovider = "ollama"\n')
        monkeypatch.setattr("openpurr.config.CONFIG_PATH", p)
        assert is_first_run() is False


# ─── load_config ──────────────────────────────────────────────────────────────


class TestLoadConfig:
    def test_returns_defaults_when_no_file(self, monkeypatch, tmp_path):
        monkeypatch.setattr("openpurr.config.CONFIG_PATH", tmp_path / "missing")
        cfg = load_config()
        assert cfg["llm"]["provider"] == DEFAULT_CONFIG["llm"]["provider"]
        assert cfg["pr"]["default_base"] == DEFAULT_CONFIG["pr"]["default_base"]

    def test_file_values_override_defaults(self, monkeypatch, tmp_path):
        p = tmp_path / ".openpurr"
        p.write_text('[llm]\nmodel = "gpt-4o"\nprovider = "openai"\n')
        monkeypatch.setattr("openpurr.config.CONFIG_PATH", p)
        cfg = load_config()
        assert cfg["llm"]["model"] == "gpt-4o"
        assert cfg["llm"]["provider"] == "openai"

    def test_unspecified_keys_keep_defaults(self, monkeypatch, tmp_path):
        p = tmp_path / ".openpurr"
        p.write_text('[pr]\ndefault_base = "develop"\n')
        monkeypatch.setattr("openpurr.config.CONFIG_PATH", p)
        cfg = load_config()
        assert cfg["pr"]["default_base"] == "develop"
        assert cfg["llm"]["provider"] == DEFAULT_CONFIG["llm"]["provider"]

    def test_unspecified_llm_keys_keep_defaults(self, monkeypatch, tmp_path):
        p = tmp_path / ".openpurr"
        p.write_text('[llm]\nmodel = "custom"\n')
        monkeypatch.setattr("openpurr.config.CONFIG_PATH", p)
        cfg = load_config()
        assert "temperature" in cfg["llm"]
        assert "keep_alive" in cfg["llm"]


# ─── write_config / round-trip ────────────────────────────────────────────────


class TestWriteConfig:
    def test_round_trip(self, monkeypatch, tmp_path):
        p = tmp_path / ".openpurr"
        monkeypatch.setattr("openpurr.config.CONFIG_PATH", p)
        data = {
            "llm": {
                "provider": "anthropic",
                "model": "claude-opus-4-8",
                "api_key": "sk-ant",
                "host": "",
                "temperature": 0.0,
                "keep_alive": "0s",
            },
            "pr": {"default_base": "develop"},
        }
        write_config(data)
        loaded = load_config()
        assert loaded["llm"]["provider"] == "anthropic"
        assert loaded["llm"]["model"] == "claude-opus-4-8"
        assert loaded["pr"]["default_base"] == "develop"

    def test_file_is_valid_toml(self, monkeypatch, tmp_path):
        p = tmp_path / ".openpurr"
        monkeypatch.setattr("openpurr.config.CONFIG_PATH", p)
        write_config(DEFAULT_CONFIG)
        with p.open("rb") as f:
            tomllib.load(f)  # must not raise


# ─── set_config_value ─────────────────────────────────────────────────────────


class TestSetConfigValue:
    def test_sets_string(self, monkeypatch, tmp_path):
        p = tmp_path / ".openpurr"
        monkeypatch.setattr("openpurr.config.CONFIG_PATH", p)
        set_config_value("model", "gpt-4o-mini")
        assert load_config()["llm"]["model"] == "gpt-4o-mini"

    def test_coerces_float(self, monkeypatch, tmp_path):
        p = tmp_path / ".openpurr"
        monkeypatch.setattr("openpurr.config.CONFIG_PATH", p)
        set_config_value("temperature", "0.7")
        assert load_config()["llm"]["temperature"] == pytest.approx(0.7)

    def test_preserves_other_values_on_repeated_set(self, monkeypatch, tmp_path):
        p = tmp_path / ".openpurr"
        monkeypatch.setattr("openpurr.config.CONFIG_PATH", p)
        set_config_value("model", "first")
        set_config_value("provider", "openai")
        cfg = load_config()
        assert cfg["llm"]["model"] == "first"
        assert cfg["llm"]["provider"] == "openai"

    def test_invalid_key_format_raises(self, monkeypatch, tmp_path):
        monkeypatch.setattr("openpurr.config.CONFIG_PATH", tmp_path / ".openpurr")
        with pytest.raises(ValueError, match="Unknown key"):
            set_config_value("badkey", "value")


# ─── get_config_value ─────────────────────────────────────────────────────────


class TestGetConfigValue:
    def test_returns_set_value(self, monkeypatch, tmp_path):
        p = tmp_path / ".openpurr"
        monkeypatch.setattr("openpurr.config.CONFIG_PATH", p)
        set_config_value("model", "my-model")
        assert get_config_value("model") == "my-model"

    def test_returns_default_for_existing_key(self, monkeypatch, tmp_path):
        monkeypatch.setattr("openpurr.config.CONFIG_PATH", tmp_path / "missing")
        assert get_config_value("base") == DEFAULT_CONFIG["pr"]["default_base"]

    def test_raises_key_error_for_unknown_key(self, monkeypatch, tmp_path):
        monkeypatch.setattr("openpurr.config.CONFIG_PATH", tmp_path / "missing")
        with pytest.raises(KeyError):
            get_config_value("llm.nonexistent_key_xyz")

    def test_raises_value_error_for_bad_format(self, monkeypatch, tmp_path):
        monkeypatch.setattr("openpurr.config.CONFIG_PATH", tmp_path / "missing")
        with pytest.raises(ValueError):
            get_config_value("nodot")


# ─── Config class ─────────────────────────────────────────────────────────────


class TestConfigClass:
    def test_all_properties(self, monkeypatch, tmp_path):
        p = tmp_path / ".openpurr"
        p.write_text(
            "[llm]\n"
            'provider = "openai"\n'
            'model = "gpt-4o"\n'
            'api_key = "sk-x"\n'
            'host = ""\n'
            "temperature = 0.2\n"
            'keep_alive = "0s"\n'
            "[pr]\n"
            'default_base = "develop"\n'
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
