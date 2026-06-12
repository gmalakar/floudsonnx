# =============================================================================
# Copyright (c) 2026 Goutam Malakar. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# =============================================================================
from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Default tasks per model_for — required by optimum (task must not be None)
MODEL_FOR_DEFAULT_TASKS: dict[str, str] = {
    "fe": "feature-extraction",
    "sc": "text-classification",
    "ranker": "text-classification",
    "s2s": "seq2seq-lm",
    "llm": "text-generation-with-past",
}


class FloudsOnnxSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="FLOUDSONNX_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Store root ────────────────────────────────────────────────────────────
    home_dir: Path = Path.home() / ".flouds"
    onnx_path: Optional[Path] = None

    # ── Session cache sizes ───────────────────────────────────────────────────
    encoder_cache_max: int = 5
    decoder_cache_max: int = 5
    seq2seq_cache_max: int = 3
    config_cache_max: int = 50

    # ── Export behaviour — defaults match flouds-export CLI defaults ──────────
    export_on_pull: bool = True
    export_optimize: bool = False
    export_optimization_level: int = 99
    export_opset: Optional[int] = None
    export_device: str = "cpu"
    export_framework: Optional[str] = None
    export_library: Optional[str] = None
    export_normalize_embeddings: bool = False
    export_trust_remote_code: bool = False
    export_use_external_data_format: bool = False
    export_use_subprocess: bool = False  # matches CLI default
    export_use_fallback_if_failed: bool = False  # matches CLI default
    export_merge: bool = False
    export_skip_validator: bool = False
    export_hf_token: Optional[str] = None

    # ── ORT provider ─────────────────────────────────────────────────────────
    session_provider: str = "CPUExecutionProvider"

    # ── HTTP server ───────────────────────────────────────────────────────────
    server_host: str = "127.0.0.1"
    server_port: int = 19720

    @field_validator("encoder_cache_max", "decoder_cache_max", "seq2seq_cache_max", "config_cache_max")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("Cache max sizes must be >= 1")
        return v

    @property
    def models_root(self) -> Path:
        root = self.onnx_path or (self.home_dir / "models")
        root.mkdir(parents=True, exist_ok=True)
        return root
