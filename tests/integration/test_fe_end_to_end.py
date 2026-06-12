# =============================================================================
# File: test_fe_end_to_end.py
# Date Created: 2026-06-10
# Date Updated: 2026-06-11
# Copyright (c) 2026 Goutam Malakar.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================
"""
Integration test: full pull → load → run for a real FE model.

Requires:
  pip install floudsonnx[export]
  do NOT install sentence-transformers

Run with:  pytest -m integration
"""
import numpy as np
import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def fe_client(tmp_path_factory):
    from floudsonnx import FloudsOnnxClient, FloudsOnnxSettings

    home = tmp_path_factory.mktemp("flouds_integration")
    return FloudsOnnxClient(FloudsOnnxSettings(home_dir=home, export_optimize=False))


def test_pull_creates_manifest(fe_client):
    manifest = fe_client.pull(
        "sentence-transformers/all-MiniLM-L6-v2",
        model_for="fe",
        task="feature-extraction",
        library="transformers",
        normalize_embeddings=True,
    )
    assert manifest.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert len(manifest.onnx_files) > 0


def test_load_returns_ready_session(fe_client):
    model = fe_client.load_model("sentence-transformers/all-MiniLM-L6-v2", model_for="fe")
    assert model.session is not None
    assert model.tokenizer is not None


def test_run_produces_embeddings(fe_client):
    model = fe_client.load_model("sentence-transformers/all-MiniLM-L6-v2", model_for="fe")
    enc = model.tokenizer(
        ["Hello world"],
        return_tensors="np",
        padding=True,
        truncation=True,
        max_length=64,
    )
    # Build feed from what the session actually expects — handles token_type_ids
    session_inputs = {inp.name for inp in model.session.get_inputs()}
    feed = {n: enc[n].astype(np.int64) for n in session_inputs if n in enc}
    for n in session_inputs - set(feed):
        feed[n] = np.zeros_like(next(iter(feed.values())))

    outputs = model.run(None, feed)
    assert outputs is not None
    assert len(outputs) > 0


def test_reload_returns_fresh_session(fe_client):
    model = fe_client.reload("sentence-transformers/all-MiniLM-L6-v2", model_for="fe")
    assert model.session is not None


def test_list_includes_pulled_model(fe_client):
    names = {m.model_name for m in fe_client.list()}
    assert "sentence-transformers/all-MiniLM-L6-v2" in names


def test_pull_with_hf_token_param(fe_client):
    manifest = fe_client.pull(
        "sentence-transformers/all-MiniLM-L6-v2",
        model_for="fe",
        hf_token=None,
    )
    assert manifest is not None
