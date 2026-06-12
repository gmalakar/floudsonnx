# =============================================================================
# Copyright (c) 2026 Goutam Malakar. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# =============================================================================
import threading
from unittest.mock import patch

import pytest

from floudsonnx.config.model_config import ModelConfig
from floudsonnx.runtime.session_pool import SessionPool


@pytest.fixture()
def model_dir(tmp_path):
    (tmp_path / "model.onnx").write_text("fake")
    return str(tmp_path)


@pytest.fixture()
def pool():
    return SessionPool(encoder_max=2, decoder_max=2, seq2seq_max=2)


class TestSessionPool:
    def test_encoder_cache_hit(self, pool, model_dir, mock_ort_session):
        cfg = ModelConfig(model_name="x", model_for="fe")
        with patch("floudsonnx.runtime.session_pool.OrtInferenceSessionLoader.load", return_value=mock_ort_session):
            s1 = pool.get_or_load_encoder(model_dir, cfg)
            s2 = pool.get_or_load_encoder(model_dir, cfg)
        assert s1 is s2

    def test_encoder_loads_once_under_concurrency(self, pool, model_dir, mock_ort_session):
        cfg = ModelConfig(model_name="x", model_for="fe")
        call_count = {"n": 0}

        def _load(*a, **kw):
            call_count["n"] += 1
            return mock_ort_session

        results = []
        with patch("floudsonnx.runtime.session_pool.OrtInferenceSessionLoader.load", side_effect=_load):

            def worker():
                results.append(pool.get_or_load_encoder(model_dir, cfg))

            threads = [threading.Thread(target=worker) for _ in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        assert call_count["n"] == 1
        assert all(r is mock_ort_session for r in results)

    def test_evict_removes_entries(self, pool, model_dir, mock_ort_session):
        cfg = ModelConfig(model_name="x", model_for="fe")
        with patch("floudsonnx.runtime.session_pool.OrtInferenceSessionLoader.load", return_value=mock_ort_session):
            pool.get_or_load_encoder(model_dir, cfg, "CPUExecutionProvider")
        assert pool.evict(model_dir) >= 1

    def test_stats_keys_present(self, pool):
        s = pool.stats()
        assert "encoder" in s and "seq2seq" in s
