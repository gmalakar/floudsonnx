# =============================================================================
# Copyright (c) 2026 Goutam Malakar. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# =============================================================================
"""
floudsonnx.runtime.strategy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
SessionStrategy enum and the resolve_strategy() decision function.

Decision table (priority order):
  encoder_only == True                         → ORT_INFERENCE_SESSION
  model_for in (fe, sc, ranker)                → ORT_INFERENCE_SESSION
  model_for == s2s AND NOT encoder_only        → ORT_SEQ2SEQ_LM
  model_for == llm AND use_seq2seqlm == True   → ORT_SEQ2SEQ_LM
  model_for == llm AND decoder_only == True    → ORT_INFERENCE_SESSION
  fallback                                     → ORT_INFERENCE_SESSION
"""
from __future__ import annotations

import logging
from enum import Enum

from floudsonnx.config.model_config import ModelConfig

log = logging.getLogger(__name__)

_ENCODER_ONLY_TYPES = {"fe", "sc", "ranker"}


class SessionStrategy(str, Enum):
    ORT_INFERENCE_SESSION = "ort_inference_session"
    ORT_SEQ2SEQ_LM = "ort_seq2seq_lm"


def resolve_strategy(config: ModelConfig) -> SessionStrategy:
    """Return the correct SessionStrategy for the given ModelConfig."""
    if config.encoder_only:
        return SessionStrategy.ORT_INFERENCE_SESSION

    mf = config.model_for.lower().strip()

    if mf in _ENCODER_ONLY_TYPES:
        return SessionStrategy.ORT_INFERENCE_SESSION

    if mf == "s2s":
        return SessionStrategy.ORT_SEQ2SEQ_LM

    if mf == "llm":
        if config.use_seq2seqlm:
            return SessionStrategy.ORT_SEQ2SEQ_LM
        return SessionStrategy.ORT_INFERENCE_SESSION

    log.warning("resolve_strategy: unknown model_for='%s'; defaulting to ORT_INFERENCE_SESSION", mf)
    return SessionStrategy.ORT_INFERENCE_SESSION
