# =============================================================================
# Copyright (c) 2026 Goutam Malakar. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# =============================================================================
from floudsonnx.config.model_config import ModelConfig
from floudsonnx.runtime.strategy import SessionStrategy, resolve_strategy


def _cfg(**kw) -> ModelConfig:
    return ModelConfig(model_name="test/model", **kw)


class TestResolveStrategy:
    def test_fe(self):
        assert resolve_strategy(_cfg(model_for="fe")) == SessionStrategy.ORT_INFERENCE_SESSION

    def test_sc(self):
        assert resolve_strategy(_cfg(model_for="sc")) == SessionStrategy.ORT_INFERENCE_SESSION

    def test_ranker(self):
        assert resolve_strategy(_cfg(model_for="ranker")) == SessionStrategy.ORT_INFERENCE_SESSION

    def test_s2s(self):
        assert resolve_strategy(_cfg(model_for="s2s")) == SessionStrategy.ORT_SEQ2SEQ_LM

    def test_s2s_encoder_only_overrides(self):
        assert resolve_strategy(_cfg(model_for="s2s", encoder_only=True)) == SessionStrategy.ORT_INFERENCE_SESSION

    def test_llm_use_seq2seqlm(self):
        assert resolve_strategy(_cfg(model_for="llm", use_seq2seqlm=True)) == SessionStrategy.ORT_SEQ2SEQ_LM

    def test_llm_decoder_only(self):
        assert resolve_strategy(_cfg(model_for="llm", decoder_only=True)) == SessionStrategy.ORT_INFERENCE_SESSION

    def test_llm_default(self):
        assert resolve_strategy(_cfg(model_for="llm")) == SessionStrategy.ORT_INFERENCE_SESSION

    def test_encoder_only_flag_always_wins(self):
        assert resolve_strategy(_cfg(model_for="s2s", encoder_only=True)) == SessionStrategy.ORT_INFERENCE_SESSION

    def test_unknown_model_for_defaults_to_ort(self):
        assert resolve_strategy(_cfg(model_for="unknown_xyz")) == SessionStrategy.ORT_INFERENCE_SESSION
