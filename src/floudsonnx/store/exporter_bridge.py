# =============================================================================
# File: exporter_bridge.py
# Date Created: 2026-06-10
# Date Updated: 2026-06-11
# Copyright (c) 2026 Goutam Malakar.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================
"""
floudsonnx.store.exporter_bridge
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Thin lazy wrapper over flouds-model-exporter 0.2.0.

Mirrors the CLI invocation:
    flouds-export export --model-for fe \\
        --model-name sentence-transformers/all-MiniLM-L6-v2 \\
        --task feature-extraction \\
        --library transformers \\
        --normalize-embeddings

All param defaults match the flouds-export CLI defaults exactly.
Key notes:
  - task must not be None; registry auto-resolves from model_for if omitted.
  - library="transformers" bypasses sentence_transformers auto-detection.
  - use_subprocess=False matches CLI default; set True for models where
    in-process export fails (e.g. sentence-transformers with ST installed).
  - ONNX_PATH env var is set to <home_dir> and restored after export.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from floudsonnx.exceptions import ExporterNotInstalledError, ExportError


class ExporterBridge:

    @staticmethod
    def export(
        model_name: str,
        model_dir: str,
        model_for: str,
        task: str,  # required — must not be None
        optimize: bool = False,
        optimization_level: int = 99,
        opset_version: int | None = None,
        device: str = "cpu",
        framework: str | None = None,
        library: str | None = None,
        normalize_embeddings: bool = False,
        force: bool = False,
        trust_remote_code: bool = False,
        use_external_data_format: bool = False,
        use_subprocess: bool = False,  # matches CLI default
        use_fallback_if_failed: bool = False,  # matches CLI default
        merge: bool = False,
        skip_validator: bool = False,
        hf_token: str | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Export *model_name* using flouds-model-exporter 0.2.0.

        Sets ONNX_PATH = <home_dir> (root above models/<model_for>/<folder>)
        before calling export(), restores it in a finally block.

        Returns the output directory path on success.
        Raises ExporterNotInstalledError or ExportError.
        """
        try:
            from model_exporter.export.pipeline import export as _export
        except ImportError as exc:
            raise ExporterNotInstalledError() from exc

        # ONNX_PATH = root above models/<model_for>/<folder>
        # model_dir  = <home>/models/<model_for>/<folder>
        # onnx_root  = <home>
        onnx_root = str(Path(model_dir).parent.parent.parent)
        _orig = os.environ.get("ONNX_PATH")
        os.environ["ONNX_PATH"] = onnx_root

        call_kwargs: dict[str, Any] = {
            "model_name": model_name,
            "model_for": model_for,
            "task": task,
            "optimize": optimize,
            "optimization_level": optimization_level,
            "device": device,
            "normalize_embeddings": normalize_embeddings,
            "force": force,
            "trust_remote_code": trust_remote_code,
            "use_external_data_format": use_external_data_format,
            "use_subprocess": use_subprocess,
            "use_fallback_if_failed": use_fallback_if_failed,
            "merge": merge,
            "skip_validator": skip_validator,
        }
        if opset_version is not None:
            call_kwargs["opset_version"] = opset_version
        if framework is not None:
            call_kwargs["framework"] = framework
        if library is not None:
            call_kwargs["library"] = library
        if hf_token is not None:
            call_kwargs["hf_token"] = hf_token

        call_kwargs.update(kwargs)

        try:
            result = _export(**call_kwargs)
        except Exception as exc:
            raise ExportError(
                f"Export failed for '{model_name}'",
                detail=str(exc),
            ) from exc
        finally:
            if _orig is None:
                os.environ.pop("ONNX_PATH", None)
            else:
                os.environ["ONNX_PATH"] = _orig

        return str(result)
