# =============================================================================
# Copyright (c) 2026 Goutam Malakar. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# =============================================================================
"""
floudsonnx.runtime.ort_seq2seq
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Loads an ORTModelForSeq2SeqLM from a local model directory.
Used for: s2s models (T5, BART, mT5) and llm with use_seq2seqlm=True.

Mirrors BaseNLPService._get_seq2seq_model():
  - ORTModelForSeq2SeqLM.from_pretrained(path, local_files_only=True)
  - Applies _supports_cache_class=False compatibility patch for optimum
    version differences (same pattern as the original service).
"""
from __future__ import annotations

import logging
from typing import Any

from floudsonnx.config.model_config import ModelConfig
from floudsonnx.exceptions import ModelLoadError, OptimumNotInstalledError

log = logging.getLogger(__name__)


class OrtSeq2SeqLoader:

    @staticmethod
    def load(model_dir: str, config: ModelConfig) -> Any:
        """
        Returns an ORTModelForSeq2SeqLM instance.
        Raises OptimumNotInstalledError or ModelLoadError.
        """
        try:
            from optimum.onnxruntime import ORTModelForSeq2SeqLM
        except ImportError as exc:
            raise OptimumNotInstalledError() from exc

        use_cache = bool(config.use_cache) if config.use_cache is not None else False
        log.debug("OrtSeq2SeqLoader: loading '%s' use_cache=%s", model_dir, use_cache)

        try:
            model = ORTModelForSeq2SeqLM.from_pretrained(
                model_dir,
                use_cache=use_cache,
                local_files_only=True,
            )
        except Exception as exc:
            raise ModelLoadError(f"ORTModelForSeq2SeqLM failed for '{model_dir}'", detail=str(exc)) from exc

        # Compatibility patch — absent in some older optimum builds
        if not hasattr(model, "_supports_cache_class"):
            model._supports_cache_class = False

        log.info("OrtSeq2SeqLoader: seq2seq model ready — %s", model_dir)
        return model
