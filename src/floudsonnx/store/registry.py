# =============================================================================
# Copyright (c) 2026 Goutam Malakar. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# =============================================================================
from __future__ import annotations

import logging
import os
import shutil
import threading
from pathlib import Path
from typing import Any, List, Optional

from floudsonnx.config.model_config import ModelConfig
from floudsonnx.config.settings import MODEL_FOR_DEFAULT_TASKS, FloudsOnnxSettings
from floudsonnx.exceptions import ManifestError
from floudsonnx.store.exporter_bridge import ExporterBridge
from floudsonnx.store.manifest import ExportOptions, ModelManifest
from floudsonnx.utils.path_guard import safe_model_folder

log = logging.getLogger(__name__)


class ModelRegistry:
    def __init__(self, settings: FloudsOnnxSettings) -> None:
        self._settings = settings
        self._export_locks: dict[str, threading.Lock] = {}
        self._lock_registry = threading.Lock()

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
        trust_remote_code: Optional[bool] = None,
        use_external_data_format: Optional[bool] = None,
        use_subprocess: Optional[bool] = None,
        use_fallback_if_failed: Optional[bool] = None,
        merge: Optional[bool] = None,
        skip_validator: Optional[bool] = None,
        hf_token: Optional[str] = None,
        **export_kwargs: Any,
    ) -> ModelManifest:
        folder = safe_model_folder(model_name)
        model_dir = self._model_dir(model_for, folder)
        key = f"{model_for}/{folder}"

        # task must not be None — optimum crashes with AttributeError otherwise
        resolved_task = task or MODEL_FOR_DEFAULT_TASKS.get(model_for.lower(), "feature-extraction")

        with self._per_model_lock(key):
            if self.exists(model_name, model_for):
                if not force:
                    log.debug("pull: '%s' already on disk; skipping export", key)
                    try:
                        return ModelManifest.load(str(model_dir))
                    except ManifestError:
                        log.warning("pull: manifest corrupt for '%s'; re-exporting", key)
                else:
                    log.info("pull: force=True; removing '%s' before re-export", key)
                    shutil.rmtree(str(model_dir), ignore_errors=True)

            os.makedirs(str(model_dir), exist_ok=True)

            s = self._settings
            eff_optimize = optimize if optimize is not None else s.export_optimize
            eff_opt_level = optimization_level if optimization_level is not None else s.export_optimization_level
            eff_opset = opset_version if opset_version is not None else s.export_opset
            eff_library = library if library is not None else s.export_library
            eff_norm = normalize_embeddings if normalize_embeddings is not None else s.export_normalize_embeddings
            eff_trust = trust_remote_code if trust_remote_code is not None else s.export_trust_remote_code
            eff_ext_data = use_external_data_format if use_external_data_format is not None else s.export_use_external_data_format
            eff_subproc = use_subprocess if use_subprocess is not None else s.export_use_subprocess
            eff_fallback = use_fallback_if_failed if use_fallback_if_failed is not None else s.export_use_fallback_if_failed
            eff_merge = merge if merge is not None else s.export_merge
            eff_skip_val = skip_validator if skip_validator is not None else s.export_skip_validator
            eff_token = hf_token or s.export_hf_token

            log.info(
                "pull: exporting '%s' (model_for=%s, task=%s, optimize=%s, library=%s, normalize_embeddings=%s)",
                model_name,
                model_for,
                resolved_task,
                eff_optimize,
                eff_library,
                eff_norm,
            )

            ExporterBridge.export(
                model_name=model_name,
                model_dir=str(model_dir),
                model_for=model_for,
                task=resolved_task,
                optimize=eff_optimize,
                optimization_level=eff_opt_level,
                opset_version=eff_opset,
                device=s.export_device,
                framework=s.export_framework,
                library=eff_library,
                normalize_embeddings=eff_norm,
                force=force,
                trust_remote_code=eff_trust,
                use_external_data_format=eff_ext_data,
                use_subprocess=bool(eff_subproc),
                use_fallback_if_failed=eff_fallback,
                merge=eff_merge,
                skip_validator=eff_skip_val,
                hf_token=eff_token,
                **export_kwargs,
            )

            manifest = self._build_manifest(model_name, model_for, folder, model_dir, resolved_task, config)
            manifest.save(str(model_dir))
            log.info("pull: manifest written for '%s'", key)
            return manifest

    def list(self) -> List[ModelManifest]:
        manifests: List[ModelManifest] = []
        models_root = self._settings.models_root
        if not models_root.exists():
            return manifests
        for model_for_dir in models_root.iterdir():
            if not model_for_dir.is_dir():
                continue
            for model_dir in model_for_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                if ModelManifest.exists(str(model_dir)):
                    try:
                        manifests.append(ModelManifest.load(str(model_dir)))
                    except ManifestError as exc:
                        log.warning("list: skipping '%s': %s", model_dir, exc)
        return manifests

    def remove(self, model_name: str, model_for: str = "fe") -> bool:
        folder = safe_model_folder(model_name)
        model_dir = self._model_dir(model_for, folder)
        key = f"{model_for}/{folder}"
        if not model_dir.exists():
            return False
        with self._per_model_lock(key):
            shutil.rmtree(str(model_dir), ignore_errors=True)
            log.info("remove: deleted '%s'", model_dir)
            return True

    def get_manifest(self, model_name: str, model_for: str = "fe") -> Optional[ModelManifest]:
        folder = safe_model_folder(model_name)
        model_dir = self._model_dir(model_for, folder)
        if not ModelManifest.exists(str(model_dir)):
            return None
        try:
            return ModelManifest.load(str(model_dir))
        except ManifestError as exc:
            log.warning("get_manifest: cannot read '%s/%s': %s", model_for, folder, exc)
            return None

    def exists(self, model_name: str, model_for: str = "fe") -> bool:
        folder = safe_model_folder(model_name)
        model_dir = self._model_dir(model_for, folder)
        return model_dir.is_dir() and ModelManifest.exists(str(model_dir))

    def model_dir_path(self, model_name: str, model_for: str = "fe") -> str:
        folder = safe_model_folder(model_name)
        return str(self._model_dir(model_for, folder))

    def _model_dir(self, model_for: str, folder: str) -> Path:
        return self._settings.models_root / model_for / folder

    def _per_model_lock(self, key: str) -> threading.Lock:
        with self._lock_registry:
            if key not in self._export_locks:
                self._export_locks[key] = threading.Lock()
            return self._export_locks[key]

    @staticmethod
    def _build_manifest(
        model_name: str,
        model_for: str,
        folder: str,
        model_dir: Path,
        task: Optional[str],
        config: Optional[ModelConfig],
    ) -> ModelManifest:
        from floudsonnx.runtime.strategy import resolve_strategy

        cfg = config or ModelConfig(model_name=model_name, model_for=model_for)
        strategy = resolve_strategy(cfg)
        manifest = ModelManifest(
            model_name=model_name,
            model_for=model_for,
            folder=folder,
            session_strategy=strategy.value,
            export_options=ExportOptions(task=task),
            config=cfg,
        )
        manifest.scan_onnx_files(str(model_dir))
        return manifest
