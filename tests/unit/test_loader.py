# =============================================================================
# Copyright (c) 2026 Goutam Malakar. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# =============================================================================
from unittest.mock import patch

import pytest

from floudsonnx.config.model_config import ModelConfig
from floudsonnx.exceptions import ModelNotFoundError
from floudsonnx.runtime.loader import LoadedModel, ModelLoader
from floudsonnx.runtime.strategy import SessionStrategy
from floudsonnx.store.manifest import ModelManifest


def _manifest(model_name="test/model", model_for="fe", strategy="ort_inference_session"):
    return ModelManifest(
        model_name=model_name,
        model_for=model_for,
        folder=model_name.split("/")[-1],
        session_strategy=strategy,
        config=ModelConfig(model_name=model_name, model_for=model_for),
    )


class TestModelLoader:
    def test_create_model_fe(self, settings, mock_ort_session, mock_tokenizer):
        loader = ModelLoader(settings)
        with (
            patch.object(loader._registry, "pull", return_value=_manifest()),
            patch.object(loader._pool, "get_or_load_encoder", return_value=mock_ort_session),
            patch.object(loader._tokenizers, "get_or_load", return_value=mock_tokenizer),
        ):
            result = loader.create_model("test/model")
        assert isinstance(result, LoadedModel)
        assert result.session is mock_ort_session
        assert result.session_strategy == SessionStrategy.ORT_INFERENCE_SESSION

    def test_load_model_raises_when_missing(self, settings):
        loader = ModelLoader(settings)
        with patch.object(loader._registry, "exists", return_value=False):
            with pytest.raises(ModelNotFoundError):
                loader.load_model("not/there")

    def test_load_model_succeeds(self, settings, mock_ort_session, mock_tokenizer):
        loader = ModelLoader(settings)
        with (
            patch.object(loader._registry, "exists", return_value=True),
            patch.object(loader._registry, "get_manifest", return_value=_manifest()),
            patch.object(loader._pool, "get_or_load_encoder", return_value=mock_ort_session),
            patch.object(loader._tokenizers, "get_or_load", return_value=mock_tokenizer),
        ):
            result = loader.load_model("test/model")
        assert result.model_name == "test/model"

    def test_reload_evicts_then_reloads(self, settings, mock_ort_session, mock_tokenizer):
        loader = ModelLoader(settings)
        with (
            patch.object(loader._registry, "exists", return_value=True),
            patch.object(loader._registry, "get_manifest", return_value=_manifest()),
            patch.object(loader._pool, "evict", return_value=1) as mock_evict,
            patch.object(loader._pool, "get_or_load_encoder", return_value=mock_ort_session),
            patch.object(loader._tokenizers, "get_or_load", return_value=mock_tokenizer),
            patch.object(loader._tokenizers, "evict"),
        ):
            result = loader.reload("test/model")
        mock_evict.assert_called_once()
        assert result.model_name == "test/model"

    def test_remove_evicts_and_deletes(self, settings):
        loader = ModelLoader(settings)
        with (
            patch.object(loader._registry, "remove", return_value=True) as mock_remove,
            patch.object(loader._pool, "evict", return_value=1),
            patch.object(loader._tokenizers, "evict"),
        ):
            assert loader.remove("test/model") is True
        mock_remove.assert_called_once()

    def test_loaded_model_run_proxy(self, mock_ort_session, mock_tokenizer):
        cfg = ModelConfig(model_name="x", model_for="fe")
        lm = LoadedModel(
            model_name="x",
            model_for="fe",
            model_dir="/tmp/x",
            config=cfg,
            tokenizer=mock_tokenizer,
            session_strategy=SessionStrategy.ORT_INFERENCE_SESSION,
            session=mock_ort_session,
        )
        lm.run(None, {"input_ids": [[1]]})
        mock_ort_session.run.assert_called_once_with(None, {"input_ids": [[1]]}, None)

    def test_loaded_model_run_raises_for_seq2seq(self, mock_seq2seq_model, mock_tokenizer):
        cfg = ModelConfig(model_name="t5", model_for="s2s")
        lm = LoadedModel(
            model_name="t5",
            model_for="s2s",
            model_dir="/tmp/t5",
            config=cfg,
            tokenizer=mock_tokenizer,
            session_strategy=SessionStrategy.ORT_SEQ2SEQ_LM,
            seq2seq_model=mock_seq2seq_model,
        )
        with pytest.raises(TypeError, match="ORTModelForSeq2SeqLM"):
            lm.run(None, {})

    def test_create_model_seq2seq(self, settings, mock_seq2seq_model, mock_tokenizer):
        loader = ModelLoader(settings)
        manifest = _manifest("t5-small", "s2s", "ort_seq2seq_lm")
        with (
            patch.object(loader._registry, "pull", return_value=manifest),
            patch.object(loader._pool, "get_or_load_seq2seq", return_value=mock_seq2seq_model),
            patch.object(loader._tokenizers, "get_or_load", return_value=mock_tokenizer),
        ):
            result = loader.create_model("t5-small", model_for="s2s")
        assert result.is_seq2seq is True
        assert result.seq2seq_model is mock_seq2seq_model
