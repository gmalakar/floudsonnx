# =============================================================================
# Copyright (c) 2026 Goutam Malakar. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# =============================================================================
from floudsonnx.config.model_config import ModelConfig


class TestModelConfig:
    def test_defaults(self):
        cfg = ModelConfig(model_name="org/model")
        assert cfg.model_for == "fe"
        assert cfg.max_length == 256
        assert cfg.pooling_strategy == "mean"
        assert cfg.encoder_only is False
        assert cfg.use_seq2seqlm is False

    def test_roundtrip_json(self):
        cfg = ModelConfig(model_name="t5-small", model_for="s2s", max_length=512)
        cfg2 = ModelConfig.model_validate(cfg.model_dump(mode="json"))
        assert cfg2.model_name == "t5-small" and cfg2.max_length == 512

    def test_input_names_defaults(self):
        cfg = ModelConfig(model_name="x")
        assert cfg.inputnames.input == "input_ids"
        assert cfg.inputnames.mask == "attention_mask"

    def test_decoder_input_names(self):
        cfg = ModelConfig(model_name="x")
        assert cfg.decoder_inputnames.encoder_output == "encoder_hidden_states"

    def test_extra_fields_allowed(self):
        cfg = ModelConfig(model_name="x", unknown_field="yes")
        assert cfg.model_name == "x"

    def test_s2s_flags(self):
        cfg = ModelConfig(model_name="t5", model_for="s2s", use_seq2seqlm=True, use_cache=True)
        assert cfg.use_seq2seqlm is True and cfg.use_cache is True
