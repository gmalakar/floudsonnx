# =============================================================================
# Copyright (c) 2026 Goutam Malakar. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# =============================================================================
from __future__ import annotations

import threading
from typing import Any, List, Optional

from floudsonnx.config.model_config import ModelConfig
from floudsonnx.config.settings import FloudsOnnxSettings
from floudsonnx.runtime.loader import LoadedModel, ModelLoader
from floudsonnx.store.manifest import ModelManifest


class FloudsOnnxClient:
    """Friendly facade over ModelLoader."""

    def __init__(self, settings: Optional[FloudsOnnxSettings] = None) -> None:
        self._loader = ModelLoader(settings)

    def pull(
        self,
        model_name: str,
        model_for: str = "fe",
        task: Optional[str] = None,
        force: bool = False,
        config: Optional[ModelConfig] = None,
        optimize: Optional[bool] = None,
        optimization_level: Optional[int] = None,
        opset_version: Optional[int] = None,
        library: Optional[str] = None,
        normalize_embeddings: Optional[bool] = None,
        trust_remote_code: bool = False,
        use_external_data_format: bool = False,
        use_subprocess: Optional[bool] = None,
        use_fallback_if_failed: bool = False,
        merge: bool = False,
        skip_validator: bool = False,
        hf_token: Optional[str] = None,
        **kwargs: Any,
    ) -> ModelManifest:
        return self._loader.pull(
            model_name,
            model_for,
            task=task,
            force=force,
            config=config,
            optimize=optimize,
            optimization_level=optimization_level,
            opset_version=opset_version,
            library=library,
            normalize_embeddings=normalize_embeddings,
            trust_remote_code=trust_remote_code,
            use_external_data_format=use_external_data_format,
            use_subprocess=use_subprocess,
            use_fallback_if_failed=use_fallback_if_failed,
            merge=merge,
            skip_validator=skip_validator,
            hf_token=hf_token,
            **kwargs,
        )

    def list(self) -> List[ModelManifest]:
        return self._loader.list()

    def remove(self, model_name: str, model_for: str = "fe") -> bool:
        return self._loader.remove(model_name, model_for)

    def create_model(
        self,
        model_name: str,
        model_for: str = "fe",
        task: Optional[str] = None,
        force_pull: bool = False,
        config: Optional[ModelConfig] = None,
        **kwargs: Any,
    ) -> LoadedModel:
        return self._loader.create_model(model_name, model_for, task=task, force_pull=force_pull, config=config, **kwargs)

    def load_model(self, model_name: str, model_for: str = "fe") -> LoadedModel:
        return self._loader.load_model(model_name, model_for)

    def reload(self, model_name: str, model_for: str = "fe") -> LoadedModel:
        return self._loader.reload(model_name, model_for)

    def unload(self, model_name: str, model_for: str = "fe") -> bool:
        return self._loader.unload(model_name, model_for)

    def is_loaded(self, model_name: str, model_for: str = "fe") -> bool:
        return self._loader.is_loaded(model_name, model_for)

    def cache_stats(self) -> dict[str, Any]:
        return self._loader.cache_stats()

    @property
    def settings(self) -> FloudsOnnxSettings:
        return self._loader._settings


_default_client: Optional[FloudsOnnxClient] = None
_default_client_lock = threading.Lock()


def get_default_client() -> FloudsOnnxClient:
    global _default_client
    if _default_client is None:
        with _default_client_lock:
            if _default_client is None:
                _default_client = FloudsOnnxClient()
    return _default_client
