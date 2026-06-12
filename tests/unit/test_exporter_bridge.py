# =============================================================================
# Copyright (c) 2026 Goutam Malakar. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# =============================================================================
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from floudsonnx.exceptions import ExporterNotInstalledError, ExportError
from floudsonnx.store.exporter_bridge import ExporterBridge


class TestExporterBridge:
    def test_raises_when_not_installed(self):
        with patch.dict(
            sys.modules,
            {
                "model_exporter": None,
                "model_exporter.export": None,
                "model_exporter.export.pipeline": None,
            },
        ):
            with pytest.raises(ExporterNotInstalledError):
                ExporterBridge.export("x/y", "/tmp/fe/x/y", "fe", None)

    def test_wraps_export_error(self, tmp_path):
        model_dir = str(tmp_path / "models" / "fe" / "mymodel")
        os.makedirs(model_dir, exist_ok=True)
        with patch.object(ExporterBridge, "export", side_effect=ExportError("Export failed for 'x/y'", detail="boom")):
            with pytest.raises(ExportError, match="Export failed"):
                ExporterBridge.export("x/y", model_dir, "fe", None)

    def test_onnx_root_is_three_levels_above_model_dir(self, tmp_path):
        model_dir = tmp_path / "models" / "fe" / "mymodel"
        model_dir.mkdir(parents=True)
        assert str(Path(str(model_dir)).parent.parent.parent) == str(tmp_path)

    def test_sets_onnx_path_env_var(self, tmp_path):
        """ExporterBridge must set ONNX_PATH to home root before calling export()."""
        model_dir = str(tmp_path / "models" / "fe" / "mymodel")
        os.makedirs(model_dir, exist_ok=True)
        captured_env = {}

        def fake_export(**kw):
            captured_env["ONNX_PATH"] = os.environ.get("ONNX_PATH")
            return model_dir

        mock_pipeline = MagicMock()
        mock_pipeline.export = fake_export

        with patch.dict(
            sys.modules,
            {
                "model_exporter": MagicMock(),
                "model_exporter.export": MagicMock(),
                "model_exporter.export.pipeline": mock_pipeline,
            },
        ):
            ExporterBridge.export("org/model", model_dir, "fe", task="feature-extraction")

        assert captured_env["ONNX_PATH"] == str(tmp_path)

    def test_restores_onnx_path_after_export(self, tmp_path):
        """ONNX_PATH env var is restored to its original value after export."""
        model_dir = str(tmp_path / "models" / "fe" / "mymodel")
        os.makedirs(model_dir, exist_ok=True)

        original = os.environ.get("ONNX_PATH")
        os.environ["ONNX_PATH"] = "/original/path"

        def fake_export(**kw):
            return model_dir

        mock_pipeline = MagicMock()
        mock_pipeline.export = fake_export

        with patch.dict(
            sys.modules,
            {
                "model_exporter": MagicMock(),
                "model_exporter.export": MagicMock(),
                "model_exporter.export.pipeline": mock_pipeline,
            },
        ):
            ExporterBridge.export("x/model", model_dir, "fe", task="feature-extraction")

        assert os.environ.get("ONNX_PATH") == "/original/path"

        if original is None:
            os.environ.pop("ONNX_PATH", None)
        else:
            os.environ["ONNX_PATH"] = original

    def test_passes_task_and_optimize_to_exporter(self, tmp_path):
        """task and optimize are forwarded to export()."""
        model_dir = str(tmp_path / "models" / "fe" / "mymodel")
        os.makedirs(model_dir, exist_ok=True)
        captured = {}

        def fake_export(**kw):
            captured.update(kw)
            return model_dir

        mock_pipeline = MagicMock()
        mock_pipeline.export = fake_export

        with patch.dict(
            sys.modules,
            {
                "model_exporter": MagicMock(),
                "model_exporter.export": MagicMock(),
                "model_exporter.export.pipeline": mock_pipeline,
            },
        ):
            ExporterBridge.export("org/model", model_dir, "fe", task="feature-extraction", optimize=True)

        assert captured.get("task") == "feature-extraction"
        assert captured.get("optimize") is True
        assert captured.get("model_name") == "org/model"
        assert captured.get("model_for") == "fe"
