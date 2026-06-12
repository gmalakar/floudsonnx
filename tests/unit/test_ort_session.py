# =============================================================================
# Copyright (c) 2026 Goutam Malakar. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# =============================================================================
from unittest.mock import MagicMock, patch

import pytest

from floudsonnx.config.model_config import ModelConfig
from floudsonnx.exceptions import ModelLoadError
from floudsonnx.runtime.ort_session import OrtInferenceSessionLoader


@pytest.fixture()
def model_dir(tmp_path):
    (tmp_path / "model_optimized.onnx").write_text("fake")
    (tmp_path / "model.onnx").write_text("fake")
    return str(tmp_path)


class TestOrtInferenceSessionLoader:
    def test_prefers_optimized_file(self, model_dir):
        cfg = ModelConfig(model_name="x", model_for="fe")
        with patch("onnxruntime.InferenceSession") as mock_sess, patch("onnxruntime.get_available_providers", return_value=["CPUExecutionProvider"]):
            mock_sess.return_value = MagicMock()
            OrtInferenceSessionLoader.load(model_dir, cfg)
            assert "optimized" in mock_sess.call_args[0][0]

    def test_falls_back_to_model_onnx(self, tmp_path):
        (tmp_path / "model.onnx").write_text("fake")
        cfg = ModelConfig(model_name="x", model_for="fe")
        with patch("onnxruntime.InferenceSession") as mock_sess, patch("onnxruntime.get_available_providers", return_value=["CPUExecutionProvider"]):
            mock_sess.return_value = MagicMock()
            OrtInferenceSessionLoader.load(str(tmp_path), cfg)
            assert mock_sess.call_args[0][0].endswith("model.onnx")

    def test_no_onnx_file_raises(self, tmp_path):
        cfg = ModelConfig(model_name="x", model_for="fe")
        with pytest.raises(ModelLoadError, match="No usable ONNX"):
            OrtInferenceSessionLoader.load(str(tmp_path), cfg)

    def test_provider_fallback(self, model_dir):
        cfg = ModelConfig(model_name="x", model_for="fe")
        with patch("onnxruntime.InferenceSession") as mock_sess, patch("onnxruntime.get_available_providers", return_value=["CPUExecutionProvider"]):
            mock_sess.return_value = MagicMock()
            OrtInferenceSessionLoader.load(model_dir, cfg, provider="CUDAExecutionProvider")
            assert mock_sess.call_args[1]["providers"] == ["CPUExecutionProvider"]

    def test_session_creation_error_wrapped(self, model_dir):
        cfg = ModelConfig(model_name="x", model_for="fe")
        with (
            patch("onnxruntime.InferenceSession", side_effect=RuntimeError("bad model")),
            patch("onnxruntime.get_available_providers", return_value=["CPUExecutionProvider"]),
        ):
            with pytest.raises(ModelLoadError, match="bad model"):
                OrtInferenceSessionLoader.load(model_dir, cfg)
