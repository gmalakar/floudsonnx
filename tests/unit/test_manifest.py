# =============================================================================
# Copyright (c) 2026 Goutam Malakar. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# =============================================================================
import pytest

from floudsonnx.config.model_config import ModelConfig
from floudsonnx.exceptions import ManifestError
from floudsonnx.store.manifest import ModelManifest


class TestModelManifest:
    def test_save_and_load_roundtrip(self, tmp_path):
        cfg = ModelConfig(model_name="test/model", model_for="fe")
        m = ModelManifest(model_name="test/model", model_for="fe", folder="model", config=cfg)
        m.save(str(tmp_path))
        loaded = ModelManifest.load(str(tmp_path))
        assert loaded.model_name == "test/model"
        assert loaded.config.model_name == "test/model"

    def test_exists_true(self, tmp_path):
        ModelManifest(model_name="x", model_for="fe", folder="x").save(str(tmp_path))
        assert ModelManifest.exists(str(tmp_path)) is True

    def test_exists_false(self, tmp_path):
        assert ModelManifest.exists(str(tmp_path)) is False

    def test_load_missing_raises(self, tmp_path):
        with pytest.raises(ManifestError, match="not found"):
            ModelManifest.load(str(tmp_path))

    def test_load_corrupt_raises(self, tmp_path):
        (tmp_path / "manifest.json").write_text("{bad json{{")
        with pytest.raises(ManifestError):
            ModelManifest.load(str(tmp_path))

    def test_scan_onnx_files(self, tmp_path):
        (tmp_path / "model.onnx").write_text("fake")
        (tmp_path / "model_optimized.onnx").write_text("fake")
        m = ModelManifest(model_name="x", model_for="fe", folder="x")
        m.scan_onnx_files(str(tmp_path))
        assert "model.onnx" in m.onnx_files
        assert "model_optimized.onnx" in m.onnx_files
