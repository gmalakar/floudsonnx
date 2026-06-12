# Contributing

Thanks for your interest in contributing to floudsonnx.

## Development setup

```powershell
# 1. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

# 2. Install runtime + dev dependencies
pip install -r requirements-dev.txt
pip install -e ".[export,seq2seq]"

# 3. Install pre-commit hooks (REQUIRED — hooks run on every commit)
pre-commit install
pre-commit install --hook-type pre-push

# 4. Verify hooks are working
pre-commit run --all-files
```

Or use the helper script:

```powershell
.\setup_dev.ps1
```

## Local checks

```powershell
pre-commit run --all-files          # format, lint, type-check
pytest tests/unit/ -q               # unit tests (fast, no network)
pytest -m integration -v            # integration tests (requires HF access)
python tools/check_dependency_sync.py
```

## Pull request checklist

- [ ] `pre-commit run --all-files` passes
- [ ] `pytest tests/unit/ -q` passes
- [ ] New behaviour has tests
- [ ] `CHANGELOG.md` updated
- [ ] No `sentence-transformers` added to dependencies
- [ ] `requirements-prod.txt` mirrors `pyproject.toml` if deps changed

## Important rules

- Do **not** install `sentence-transformers` — always pass `library="transformers"` explicitly
- `flouds-model-exporter` must stay in `[export]` optional dep only
- All export param defaults must match `flouds-export` CLI defaults exactly

## Reporting issues

Include: OS, Python version, exact command, full error output, steps to reproduce.
