# =============================================================================
# Copyright (c) 2026 Goutam Malakar. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# =============================================================================
from unittest.mock import MagicMock, patch

from floudsonnx.api.client import FloudsOnnxClient, get_default_client
from floudsonnx.config.model_config import ModelConfig
from floudsonnx.runtime.loader import LoadedModel
from floudsonnx.runtime.strategy import SessionStrategy
from floudsonnx.store.manifest import ModelManifest


def _loaded(model_name="x", model_for="fe"):
    cfg = ModelConfig(model_name=model_name, model_for=model_for)
    return LoadedModel(
        model_name=model_name,
        model_for=model_for,
        model_dir="/tmp/x",
        config=cfg,
        tokenizer=MagicMock(),
        session_strategy=SessionStrategy.ORT_INFERENCE_SESSION,
        session=MagicMock(),
    )


class TestFloudsOnnxClient:
    def test_create_model_delegates(self, settings):
        client = FloudsOnnxClient(settings)
        loaded = _loaded()
        with patch.object(client._loader, "create_model", return_value=loaded) as mock:
            result = client.create_model("x")
        mock.assert_called_once_with("x", "fe", task=None, force_pull=False, config=None)
        assert result is loaded

    def test_load_model_delegates(self, settings):
        client = FloudsOnnxClient(settings)
        loaded = _loaded()
        with patch.object(client._loader, "load_model", return_value=loaded):
            assert client.load_model("x") is loaded

    def test_remove_delegates(self, settings):
        client = FloudsOnnxClient(settings)
        with patch.object(client._loader, "remove", return_value=True):
            assert client.remove("x") is True

    def test_list_delegates(self, settings):
        client = FloudsOnnxClient(settings)
        manifests = [ModelManifest(model_name="x", model_for="fe", folder="x")]
        with patch.object(client._loader, "list", return_value=manifests):
            assert client.list() == manifests

    def test_pull_forwards_hf_token(self, settings):
        client = FloudsOnnxClient(settings)
        manifest = ModelManifest(model_name="x", model_for="fe", folder="x")
        with patch.object(client._loader, "pull", return_value=manifest) as mock_pull:
            client.pull("x", hf_token="hf_abc123")
        call_kwargs = mock_pull.call_args[1]
        assert call_kwargs.get("hf_token") == "hf_abc123"

    def test_get_default_client_singleton(self):
        assert get_default_client() is get_default_client()
