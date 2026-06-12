# =============================================================================
# Copyright (c) 2026 Goutam Malakar. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# =============================================================================
from unittest.mock import MagicMock, patch

import pytest

from floudsonnx.config.model_config import ModelConfig
from floudsonnx.exceptions import ModelLoadError, OptimumNotInstalledError
from floudsonnx.runtime.ort_seq2seq import OrtSeq2SeqLoader


@pytest.fixture()
def s2s_dir(tmp_path):
    (tmp_path / "encoder_model.onnx").write_text("fake")
    (tmp_path / "decoder_model.onnx").write_text("fake")
    return str(tmp_path)


@pytest.fixture()
def s2s_config():
    return ModelConfig(model_name="t5-small", model_for="s2s")


class TestOrtSeq2SeqLoader:
    def test_raises_when_optimum_not_installed(self, s2s_dir, s2s_config):
        with patch.dict(
            "sys.modules",
            {
                "optimum": None,
                "optimum.onnxruntime": None,
            },
        ):
            with pytest.raises(OptimumNotInstalledError):
                OrtSeq2SeqLoader.load(s2s_dir, s2s_config)

    def test_returns_model_on_success(self, s2s_dir, s2s_config):
        mock_model = MagicMock()
        mock_cls = MagicMock()
        mock_cls.from_pretrained.return_value = mock_model

        with patch.dict("sys.modules", {"optimum": MagicMock(), "optimum.onnxruntime": MagicMock()}):
            with patch("floudsonnx.runtime.ort_seq2seq.OrtSeq2SeqLoader.load", return_value=mock_model):
                result = OrtSeq2SeqLoader.load(s2s_dir, s2s_config)

        assert result is mock_model

    def test_applies_supports_cache_class_patch(self, s2s_dir, s2s_config):
        """Model missing _supports_cache_class gets it patched to False."""
        mock_model = MagicMock(spec=[])  # spec=[] means no attributes by default
        mock_cls = MagicMock()
        mock_cls.from_pretrained.return_value = mock_model

        mock_optimum_ort = MagicMock()
        mock_optimum_ort.ORTModelForSeq2SeqLM = mock_cls

        with patch.dict(
            "sys.modules",
            {
                "optimum": MagicMock(),
                "optimum.onnxruntime": mock_optimum_ort,
            },
        ):
            result = OrtSeq2SeqLoader.load(s2s_dir, s2s_config)

        assert result._supports_cache_class is False

    def test_does_not_overwrite_existing_supports_cache_class(self, s2s_dir, s2s_config):
        """Model that already has _supports_cache_class=True keeps it."""
        mock_model = MagicMock()
        mock_model._supports_cache_class = True
        mock_cls = MagicMock()
        mock_cls.from_pretrained.return_value = mock_model

        mock_optimum_ort = MagicMock()
        mock_optimum_ort.ORTModelForSeq2SeqLM = mock_cls

        with patch.dict(
            "sys.modules",
            {
                "optimum": MagicMock(),
                "optimum.onnxruntime": mock_optimum_ort,
            },
        ):
            result = OrtSeq2SeqLoader.load(s2s_dir, s2s_config)

        assert result._supports_cache_class is True

    def test_use_cache_false_by_default(self, s2s_dir):
        cfg = ModelConfig(model_name="t5", model_for="s2s")  # use_cache=None
        mock_cls = MagicMock()
        mock_cls.from_pretrained.return_value = MagicMock()
        mock_optimum_ort = MagicMock()
        mock_optimum_ort.ORTModelForSeq2SeqLM = mock_cls

        with patch.dict(
            "sys.modules",
            {
                "optimum": MagicMock(),
                "optimum.onnxruntime": mock_optimum_ort,
            },
        ):
            OrtSeq2SeqLoader.load(s2s_dir, cfg)

        call_kwargs = mock_cls.from_pretrained.call_args[1]
        assert call_kwargs["use_cache"] is False

    def test_use_cache_true_when_config_set(self, s2s_dir):
        cfg = ModelConfig(model_name="t5", model_for="s2s", use_cache=True)
        mock_cls = MagicMock()
        mock_cls.from_pretrained.return_value = MagicMock()
        mock_optimum_ort = MagicMock()
        mock_optimum_ort.ORTModelForSeq2SeqLM = mock_cls

        with patch.dict(
            "sys.modules",
            {
                "optimum": MagicMock(),
                "optimum.onnxruntime": mock_optimum_ort,
            },
        ):
            OrtSeq2SeqLoader.load(s2s_dir, cfg)

        call_kwargs = mock_cls.from_pretrained.call_args[1]
        assert call_kwargs["use_cache"] is True

    def test_wraps_load_exception_as_model_load_error(self, s2s_dir, s2s_config):
        mock_cls = MagicMock()
        mock_cls.from_pretrained.side_effect = RuntimeError("corrupt model")
        mock_optimum_ort = MagicMock()
        mock_optimum_ort.ORTModelForSeq2SeqLM = mock_cls

        with patch.dict(
            "sys.modules",
            {
                "optimum": MagicMock(),
                "optimum.onnxruntime": mock_optimum_ort,
            },
        ):
            with pytest.raises(ModelLoadError, match="corrupt model"):
                OrtSeq2SeqLoader.load(s2s_dir, s2s_config)
