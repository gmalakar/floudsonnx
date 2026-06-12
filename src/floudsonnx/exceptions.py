# =============================================================================
# Copyright (c) 2026 Goutam Malakar. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# =============================================================================
"""
floudsonnx.exceptions
~~~~~~~~~~~~~~~~~~~~~
Full exception hierarchy. All public exceptions importable from the package root.
"""


class FloudsOnnxError(Exception):
    """Base exception for all floudsonnx errors."""

    def __init__(self, message: str, *, detail: str | None = None):
        super().__init__(message)
        self.detail = detail

    def __str__(self) -> str:
        if self.detail:
            return f"{super().__str__()} — {self.detail}"
        return super().__str__()


# ── Store layer ───────────────────────────────────────────────────────────────


class ModelNotFoundError(FloudsOnnxError):
    """Model does not exist in the local store and cannot be pulled."""


class ModelAlreadyExistsError(FloudsOnnxError):
    """pull() called without force=True on an already-stored model."""


class ManifestError(FloudsOnnxError):
    """manifest.json is missing, corrupt, or fails schema validation."""


class ExportError(FloudsOnnxError):
    """flouds-model-exporter raised an error during ONNX export."""


class ExporterNotInstalledError(ExportError):
    """flouds-model-exporter is not installed."""

    def __init__(self) -> None:
        super().__init__("flouds-model-exporter is required for auto-export. " "Install it with: pip install floudsonnx[export]")


# ── Runtime layer ─────────────────────────────────────────────────────────────


class ModelLoadError(FloudsOnnxError):
    """ORT InferenceSession or ORTModelForSeq2SeqLM failed to load."""


class TokenizerError(FloudsOnnxError):
    """Tokenizer could not be loaded from disk."""


class StrategyError(FloudsOnnxError):
    """Cannot determine a session strategy from the given ModelConfig."""


class CacheError(FloudsOnnxError):
    """Internal session pool or tokenizer cache error."""


class OptimumNotInstalledError(ModelLoadError):
    """optimum[onnxruntime] is not installed; required for seq2seq models."""

    def __init__(self) -> None:
        super().__init__("optimum[onnxruntime] is required for seq2seq / LLM model loading. " "Install it with: pip install floudsonnx[seq2seq]")


# ── Server layer ──────────────────────────────────────────────────────────────


class ServerNotInstalledError(FloudsOnnxError):
    """fastapi / uvicorn are not installed; required for the HTTP server."""

    def __init__(self) -> None:
        super().__init__("fastapi and uvicorn are required for the HTTP server. " "Install them with: pip install floudsonnx[server]")
