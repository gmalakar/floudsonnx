# =============================================================================
# Copyright (c) 2026 Goutam Malakar. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# =============================================================================
from unittest.mock import patch

from floudsonnx.store.manifest import ModelManifest
from floudsonnx.store.registry import ModelRegistry


class TestModelRegistry:
    def test_exists_false_when_empty(self, settings):
        assert ModelRegistry(settings).exists("sentence-transformers/all-MiniLM-L6-v2") is False

    def test_pull_calls_exporter_when_missing(self, settings):
        reg = ModelRegistry(settings)
        with (
            patch("floudsonnx.store.registry.ExporterBridge.export", return_value="/fake") as mock_exp,
            patch.object(ModelManifest, "scan_onnx_files"),
        ):
            reg.pull("test/model", model_for="fe", task="feature-extraction")
        mock_exp.assert_called_once()

    def test_pull_passes_correct_onnx_root(self, settings):
        """onnx_path passed to exporter must be home_dir, not models_root."""
        reg = ModelRegistry(settings)
        captured = {}

        def fake_export(**kw):
            captured.update(kw)
            model_dir = kw.get("model_dir", "")
            import os

            os.makedirs(model_dir, exist_ok=True)
            ModelManifest(model_name="test/model", model_for="fe", folder="model").save(model_dir)
            return model_dir

        with patch("floudsonnx.store.registry.ExporterBridge.export", side_effect=fake_export), patch.object(ModelManifest, "scan_onnx_files"):
            reg.pull("test/model", model_for="fe")

        expected_model_dir = str(settings.models_root / "fe" / "model")
        assert captured["model_dir"] == expected_model_dir

    def test_pull_skips_export_when_exists(self, settings):
        reg = ModelRegistry(settings)
        d = settings.models_root / "fe" / "model"
        d.mkdir(parents=True)
        ModelManifest(model_name="test/model", model_for="fe", folder="model").save(str(d))
        with patch("floudsonnx.store.registry.ExporterBridge.export") as mock_exp:
            reg.pull("test/model", model_for="fe")
        mock_exp.assert_not_called()

    def test_pull_force_re_exports(self, settings):
        reg = ModelRegistry(settings)
        d = settings.models_root / "fe" / "model"
        d.mkdir(parents=True)
        ModelManifest(model_name="test/model", model_for="fe", folder="model").save(str(d))
        with (
            patch("floudsonnx.store.registry.ExporterBridge.export", return_value=str(d)) as mock_exp,
            patch.object(ModelManifest, "scan_onnx_files"),
        ):
            reg.pull("test/model", model_for="fe", force=True)
        mock_exp.assert_called_once()

    def test_pull_forwards_hf_token(self, settings):
        reg = ModelRegistry(settings)
        captured = {}

        def fake_export(**kw):
            captured.update(kw)
            import os

            os.makedirs(kw["model_dir"], exist_ok=True)
            ModelManifest(model_name="test/model", model_for="fe", folder="model").save(kw["model_dir"])
            return kw["model_dir"]

        with patch("floudsonnx.store.registry.ExporterBridge.export", side_effect=fake_export), patch.object(ModelManifest, "scan_onnx_files"):
            reg.pull("test/model", model_for="fe", hf_token="hf_test_token")
        assert captured.get("hf_token") == "hf_test_token"

    def test_list_returns_manifests(self, settings):
        reg = ModelRegistry(settings)
        for folder in ["model-a", "model-b"]:
            d = settings.models_root / "fe" / folder
            d.mkdir(parents=True)
            ModelManifest(model_name=f"org/{folder}", model_for="fe", folder=folder).save(str(d))
        names = {m.model_name for m in reg.list()}
        assert "org/model-a" in names and "org/model-b" in names

    def test_remove_deletes_directory(self, settings):
        reg = ModelRegistry(settings)
        d = settings.models_root / "fe" / "model"
        d.mkdir(parents=True)
        ModelManifest(model_name="test/model", model_for="fe", folder="model").save(str(d))
        assert reg.remove("test/model", "fe") is True
        assert not d.exists()

    def test_remove_nonexistent_returns_false(self, settings):
        assert ModelRegistry(settings).remove("no/such", "fe") is False

    def test_concurrent_pull_calls_exporter_once(self, settings):
        import threading

        reg = ModelRegistry(settings)
        call_count = {"n": 0}

        def fake_export(**kw):
            call_count["n"] += 1
            import os

            os.makedirs(kw["model_dir"], exist_ok=True)
            ModelManifest(model_name="test/model", model_for="fe", folder="model").save(kw["model_dir"])
            return kw["model_dir"]

        errors = []

        def do_pull():
            try:
                with (
                    patch("floudsonnx.store.registry.ExporterBridge.export", side_effect=fake_export),
                    patch.object(ModelManifest, "scan_onnx_files"),
                ):
                    reg.pull("test/model", "fe")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=do_pull) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors and call_count["n"] == 1
