# =============================================================================
# Copyright (c) 2026 Goutam Malakar. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# =============================================================================
import threading
from unittest.mock import MagicMock, patch

import pytest

from floudsonnx.exceptions import TokenizerError
from floudsonnx.runtime.tokenizer_cache import TokenizerCache


class TestTokenizerCache:
    def test_returns_cached_on_second_call(self, tmp_path):
        tok = MagicMock()
        cache = TokenizerCache()
        with patch("transformers.AutoTokenizer.from_pretrained", return_value=tok) as mock_load:
            r1 = cache.get_or_load(str(tmp_path))
            r2 = cache.get_or_load(str(tmp_path))
        assert r1 is r2 and mock_load.call_count == 1

    def test_thread_local_isolation(self, tmp_path):
        """Each thread gets its own tokenizer instance from thread-local storage."""
        cache = TokenizerCache()
        results = {}
        errors = []

        # Patch at test level so both threads see it safely
        with patch("transformers.AutoTokenizer.from_pretrained", side_effect=lambda *a, **kw: MagicMock()):

            def load_in_thread(name):
                try:
                    results[name] = cache.get_or_load(str(tmp_path))
                except Exception as e:
                    errors.append(e)

            t1 = threading.Thread(target=load_in_thread, args=("a",))
            t2 = threading.Thread(target=load_in_thread, args=("b",))
            t1.start()
            t2.start()
            t1.join()
            t2.join()

        assert not errors
        assert "a" in results and "b" in results
        # Thread-local: each thread loaded its own distinct instance
        assert results["a"] is not results["b"]

    def test_legacy_fallback_on_first_failure(self, tmp_path):
        good_tok = MagicMock()
        call_kwargs = []

        def fake_load(path, **kw):
            call_kwargs.append(kw)
            if kw.get("use_fast", True):
                raise OSError("broken fast tokenizer")
            return good_tok

        cache = TokenizerCache()
        with patch("transformers.AutoTokenizer.from_pretrained", side_effect=fake_load):
            result = cache.get_or_load(str(tmp_path))
        assert result is good_tok
        assert any(not kw.get("use_fast", True) for kw in call_kwargs)

    def test_raises_tokenizer_error_when_all_fail(self, tmp_path):
        def always_fail(path, **kw):
            raise OSError("always fails")

        cache = TokenizerCache()
        with patch("transformers.AutoTokenizer.from_pretrained", side_effect=always_fail):
            with pytest.raises(TokenizerError):
                cache.get_or_load(str(tmp_path))

    def test_evict_forces_reload(self, tmp_path):
        tok = MagicMock()
        cache = TokenizerCache()
        with patch("transformers.AutoTokenizer.from_pretrained", return_value=tok):
            cache.get_or_load(str(tmp_path))
        cache.evict(str(tmp_path))
        with patch("transformers.AutoTokenizer.from_pretrained", return_value=tok) as mock_load:
            cache.get_or_load(str(tmp_path))
        assert mock_load.call_count == 1
