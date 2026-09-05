# AGENTS.md

## Commands

- Install deps: `uv sync --dev` (uses `dependency-groups.dev` in `pyproject.toml:33`; `uv.lock` is gitignored — not committed)
- Run all tests: `uv run pytest` — CI matrix is 3.11/3.12/3.13/3.14 (`.github/workflows/ci.yml:15`)
- Single file/suite/case: `uv run pytest tests/test_config.py` / `uv run pytest tests/test_config.py -k TestParseEnv` / `uv run pytest tests/test_config.py::TestParseEnv::test_basic_key_value`
- Build check: `python -m build && twine check dist/*` (publish flow in `.github/workflows/publish.yml:20`)

No lint/typecheck/formatter config in repo — don't add one unless requested.

## Style — Docstrings

- All scripts in `src/openpurr/**/*.py` must have Google Style docstrings — modules, classes, functions, methods, and properties. Verified with `pydocstyle --convention=google src/openpurr` (zero warnings). Keep Args/Returns/Raises sections consistent; properties use a one-line description, functions with args use full Google sections.

## Structure

- Single package, `src` layout: `src/openpurr/` → `openpurr.*` (`pyproject.toml:31`). Entry points `opo`/`openpurr` → `openpurr.main:app` (`pyproject.toml:23`).
- CLI: `src/openpurr/main.py` (Typer app, `no_args_is_help=False` — bare `opo` runs PR generation). Subcommands `review`/`setup`/`models`/`config {describe,get,set}`.
- Config: `src/openpurr/config.py` — `~/.openpurr` (`CONFIG_PATH = Path.home()/".openpurr"`). LLM factory: `src/openpurr/llm.py:14`. System prompts: `src/openpurr/pr.py:23`. Git diff & filtering: `src/openpurr/utils/git.py:8`. Setup wizard: `src/openpurr/setup_wizard.py`. Model discovery: `src/openpurr/model_catalog.py:87`.

## Config File Gotchas — `~/.openpurr`

- Format is flat `OPO_KEY=value` lines, no quoting/sections/TOML (`src/openpurr/config.py:1`). Everything after a lone `---` line is free-text prompt overrides — never parsed as env (`_split_config_text:126`).
- Prompt headers are `INIT PROMPT:` / `REVIEW PROMPT:` (case-insensitive, `PROMPT_HEADERS:86`). `write_config()` preserves existing prompts via `load_prompts()` — `opo config set` never clobbers them (`config.py:159`).
- Isolated construction: `Config(data={...})` does **not** auto-load prompts from disk unless `prompts` is also passed (`config.py:214`). Tests must monkeypatch `openpurr.config.CONFIG_PATH` to a `tmp_path` file — never write to real `~/.openpurr` (`tests/test_config.py:76`).
- Short keys map via `SHORT_KEY_MAP` (`config.py:29`): `provider/model/api_key/host/temperature/keep_alive/base/language` → `OPO_*`. Unknown keys raise `ValueError`.
- `resolve_base_url` (`config.py:100`): explicit non-default `host` wins; otherwise `PROVIDER_BASE_URLS` (gemini/openrouter/deepseek/llamacpp/mlx). Default `OPO_HOST` (`http://localhost:11434`) does **not** count as custom.

## Git & Providers

- Diff extraction filters files containing `EXCLUDED_PATTERNS` (`utils/git.py:8`): `.lock`, `lock.json`, `pnpm-lock.yaml`, `package-lock.json`, `yarn.lock`, `.min.js`, `.svg`, `.map` — substring match (`_is_excluded:45`).
- `get_diff` uses `base...HEAD` three-dot diff; `get_recent_commits_diff` uses `HEAD~N..HEAD`.
- Rename-aware diff: `get_diff`/`get_recent_commits_diff` use `git diff --name-status` and `_parse_name_status` (`utils/git.py:49`). Passing only the new path (e.g. `--name-only` → `git diff -- new`) makes git render a pure rename as `new file`/`/dev/null` — LLM then hallucinates "add/introduce module". Fix is to pass **both** old+new paths to `git diff --` so the diff keeps `rename from`/`rename to` with `similarity index 100%` headers.
- `model_catalog.list_models()` never raises — returns `[]` on any failure. `openai`/`gemini` names are filtered by `_NON_CHAT_DENYLIST` (`model_catalog.py:19`). Callers fall back to manual model entry.

## Testing Notes

- `tool.pytest.ini_options.testpaths = ["tests"]` (`pyproject.toml:36`). Tests mock `CONFIG_PATH` and provider SDKs; no live LLM/git needed.
- Requires Python `>=3.11` (`pyproject.toml:8`).
