# =============================================================================
# Copyright (c) 2026 Goutam Malakar. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# =============================================================================
"""
floudsonnx.runtime.ort_session
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Loads a raw onnxruntime.InferenceSession.
Used for: fe, sc, ranker, decoder-only llm.

ONNX file selection priority:
  1. <model_dir>/<config.optimized_onnx_model>   (model_optimized.onnx)
  2. <model_dir>/<config.encoder_onnx_model>     (model.onnx)
  3. <model_dir>/model.onnx                      (canonical fallback)

Provider validation: falls back to CPUExecutionProvider if requested
provider is unavailable — mirrors BaseNLPService._get_encoder_session().
"""
from __future__ import annotations

import logging
import os
from typing import List

import onnxruntime as ort

from floudsonnx.config.model_config import ModelConfig
from floudsonnx.exceptions import ModelLoadError
from floudsonnx.utils.path_guard import safe_join

log = logging.getLogger(__name__)

_FALLBACK_PROVIDER = "CPUExecutionProvider"
_CANDIDATE_NAMES = ["model_optimized.onnx", "model.onnx"]


class OrtInferenceSessionLoader:

    @staticmethod
    def load(
        model_dir: str,
        config: ModelConfig,
        provider: str = _FALLBACK_PROVIDER,
    ) -> ort.InferenceSession:
        onnx_path = OrtInferenceSessionLoader._resolve_onnx_file(model_dir, config)
        resolved_provider = OrtInferenceSessionLoader._resolve_provider(provider)
        log.debug("OrtInferenceSessionLoader: loading '%s' provider=%s", onnx_path, resolved_provider)
        try:
            session = ort.InferenceSession(onnx_path, providers=[resolved_provider])
        except Exception as exc:
            raise ModelLoadError(f"ort.InferenceSession failed for '{onnx_path}'", detail=str(exc)) from exc
        log.info("OrtInferenceSessionLoader: session ready — %s", os.path.basename(onnx_path))
        return session

    @staticmethod
    def _resolve_onnx_file(model_dir: str, config: ModelConfig) -> str:
        candidates = [config.optimized_onnx_model, config.encoder_onnx_model] + _CANDIDATE_NAMES
        seen: set[str] = set()
        ordered: List[str] = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                ordered.append(c)
        for name in ordered:
            try:
                path = safe_join(model_dir, name)
            except ValueError:
                continue
            if os.path.isfile(path):
                return path
        raise ModelLoadError(f"No usable ONNX file found in '{model_dir}'", detail=f"Tried: {ordered}")

    @staticmethod
    def _resolve_provider(requested: str) -> str:
        available = ort.get_available_providers()
        if requested in available:
            return requested
        log.warning("Provider '%s' not available (available: %s); falling back to %s", requested, available, _FALLBACK_PROVIDER)
        return _FALLBACK_PROVIDER
