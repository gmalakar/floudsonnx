# =============================================================================
# Copyright (c) 2026 Goutam Malakar. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# =============================================================================
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from floudsonnx.config.model_config import ModelConfig
from floudsonnx.config.settings import FloudsOnnxSettings


@pytest.fixture()
def tmp_home(tmp_path: Path) -> Path:
    home = tmp_path / ".flouds"
    home.mkdir()
    return home


@pytest.fixture()
def settings(tmp_home: Path) -> FloudsOnnxSettings:
    return FloudsOnnxSettings(home_dir=tmp_home)


@pytest.fixture()
def fe_config() -> ModelConfig:
    return ModelConfig(model_name="sentence-transformers/all-MiniLM-L6-v2", model_for="fe")


@pytest.fixture()
def s2s_config() -> ModelConfig:
    return ModelConfig(model_name="t5-small", model_for="s2s")


@pytest.fixture()
def mock_ort_session() -> MagicMock:
    session = MagicMock()
    session.run.return_value = [[[0.1] * 384]]
    return session


@pytest.fixture()
def mock_seq2seq_model() -> MagicMock:
    model = MagicMock()
    model.generate.return_value = [[1, 2, 3]]
    model._supports_cache_class = False
    return model


@pytest.fixture()
def mock_tokenizer() -> MagicMock:
    tok = MagicMock()
    tok.return_value = {"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}
    return tok
