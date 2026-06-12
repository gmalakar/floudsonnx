# Changelog

All notable changes to `floudsonnx` will be documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
Versioning: [Semantic Versioning](https://semver.org/)

---

## [0.1.0] - 2026-06-11

### Added
- `FloudsOnnxClient` — primary API facade; process-level singleton via `get_default_client()`
- `create_model()`, `load_model()`, `pull()`, `list_models()`, `remove_model()` top-level convenience functions
- `ModelLoader` — orchestrates `ModelRegistry` + `SessionPool` + `TokenizerCache`
- `ModelRegistry` — pull / list / remove with per-model `threading.Lock`; idempotent pull
- `ExporterBridge` — lazy wrapper over `flouds-model-exporter 0.2.0`; all params aligned with PyPI API; sets `ONNX_PATH` env var before calling `export()`
- `MODEL_FOR_DEFAULT_TASKS` — auto-resolves `task` from `model_for` so `optimum` never receives `task=None`
- `SessionStrategy` — `ORT_INFERENCE_SESSION` / `ORT_SEQ2SEQ_LM` with `resolve_strategy()` decision table
- `OrtInferenceSessionLoader` — raw `ort.InferenceSession`; optimized → canonical fallback; provider validation
- `OrtSeq2SeqLoader` — `ORTModelForSeq2SeqLM`; `_supports_cache_class` compatibility patch
- `SessionPool` — thread-safe LRU caches (encoder / decoder / seq2seq) with double-checked locking
- `TokenizerCache` — thread-local; 3-attempt fallback (fast → legacy → HF Hub)
- `ModelManifest` — per-model `manifest.json` schema; self-describing store
- `FloudsOnnxSettings` — full `pydantic-settings` config; `FLOUDSONNX_` env prefix; all export params match CLI defaults
- `ModelConfig` — self-contained replacement for `OnnxConfig`; Pydantic v2; all fields from original
- `LoadedModel` — returned to all callers; `.run()` proxy for encoder models; `.seq2seq_model` for S2S/LLM
- CLI — `floudsonnx pull / list / info / remove / reload / stats / serve`; `--library`, `--normalize-embeddings` flags
- Optional FastAPI HTTP server (`floudsonnx[server]`)
- `manual_test.py` — 9-step smoke test; auto-builds `input_feed` from session's expected inputs
- `test_all_models.py` — unified test covering fe / s2s / ranker / llm with per-type inference
- `examples/` — `test_fe.py`, `test_s2s.py`, `test_ranker.py`, `test_llm.py`
- 82+ unit tests; CI matrix ubuntu + windows × py3.11 + py3.12
- `.pre-commit-config.yaml` — black, isort, flake8, mypy, bandit, pyright, add-header, dependency-sync, pre-push tests
- `tools/add_all_headers.py`, `tools/pre_push_checks.py`, `tools/check_dependency_sync.py`, `tools/bump_version.py`
- `requirements-prod.txt`, `requirements-dev.txt`, `requirements-export.txt`, `requirements-seq2seq.txt`

### Architecture decisions
- `sentence-transformers` intentionally excluded from all dependency files
- `flouds-model-exporter` is optional (`[export]` extra only)
- `library="transformers"` defaulted for `fe/sc/ranker` in CLI and test scripts to bypass ST auto-detection
- `task` auto-resolved from `model_for` in `ModelRegistry.pull()` — prevents `optimum` `AttributeError`
- `use_subprocess=False` default matches `flouds-export` CLI default exactly

### Dependencies
- Core: `onnxruntime>=1.20.1`, `transformers>=4.44.0,<4.58.0`, `pydantic>=2.0`, `pydantic-settings>=2.0`, `numpy>=1.26.4,<1.27`
- `[export]`: `flouds-model-exporter>=0.2.0`
- `[seq2seq]`: `optimum[onnxruntime]>=1.22.0`
- `[server]`: `fastapi>=0.110.0`, `uvicorn[standard]>=0.29.0`
