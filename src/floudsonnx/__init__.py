# =============================================================================
# Copyright (c) 2026 Goutam Malakar. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# =============================================================================
"""
floudsonnx
~~~~~~~~~~
Ollama-style ONNX model store and runtime.

Quick start:
    from floudsonnx import create_model

    model = create_model("sentence-transformers/all-MiniLM-L6-v2")
    outputs = model.run(None, {"input_ids": ids, "attention_mask": mask})
"""
from typing import Any, List

from floudsonnx.api.client import FloudsOnnxClient, get_default_client
from floudsonnx.config.model_config import ModelConfig
from floudsonnx.config.settings import FloudsOnnxSettings
from floudsonnx.exceptions import (
    ExporterNotInstalledError,
    ExportError,
    FloudsOnnxError,
    ManifestError,
    ModelLoadError,
    ModelNotFoundError,
    OptimumNotInstalledError,
    StrategyError,
    TokenizerError,
)
from floudsonnx.runtime.loader import LoadedModel
from floudsonnx.runtime.strategy import SessionStrategy
from floudsonnx.store.manifest import ModelManifest

__version__ = "0.1.0"
__all__ = [
    # Primary API
    "create_model",
    "load_model",
    "pull",
    "list_models",
    "remove_model",
    # Client / settings
    "FloudsOnnxClient",
    "FloudsOnnxSettings",
    "get_default_client",
    # Types
    "ModelConfig",
    "LoadedModel",
    "SessionStrategy",
    # Exceptions
    "FloudsOnnxError",
    "ModelNotFoundError",
    "ModelLoadError",
    "TokenizerError",
    "ExportError",
    "ExporterNotInstalledError",
    "OptimumNotInstalledError",
    "ManifestError",
    "StrategyError",
]


# ── Top-level convenience functions (delegate to default client singleton) ────


def create_model(model_name: str, model_for: str = "fe", **kwargs: Any) -> LoadedModel:
    """Pull (auto-export if missing) + load session. Primary entry-point."""
    return get_default_client().create_model(model_name, model_for, **kwargs)


def load_model(model_name: str, model_for: str = "fe") -> LoadedModel:
    """Load a model already on disk (no auto-pull)."""
    return get_default_client().load_model(model_name, model_for)


def pull(model_name: str, model_for: str = "fe", **kwargs: Any) -> ModelManifest:
    """Export model to disk only (no session load)."""
    return get_default_client().pull(model_name, model_for, **kwargs)


def list_models() -> List[ModelManifest]:
    """Return manifests for all locally stored models."""
    return get_default_client().list()


def remove_model(model_name: str, model_for: str = "fe") -> bool:
    """Delete model from disk and evict from session pool."""
    return get_default_client().remove(model_name, model_for)
