# =============================================================================
# Copyright (c) 2026 Goutam Malakar. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# =============================================================================
import threading

from floudsonnx.utils.concurrent_dict import ConcurrentDict


class TestConcurrentDict:
    def test_put_and_get(self):
        d: ConcurrentDict[str] = ConcurrentDict("test", 3)
        d.put("k1", "v1")
        assert d.get("k1") == "v1"

    def test_miss_returns_none(self):
        d: ConcurrentDict[str] = ConcurrentDict("test", 3)
        assert d.get("missing") is None

    def test_lru_eviction(self):
        d: ConcurrentDict[int] = ConcurrentDict("test", 2)
        d.put("a", 1)
        d.put("b", 2)
        d.put("c", 3)
        assert d.get("a") is None
        assert d.get("b") == 2 and d.get("c") == 3

    def test_lru_access_updates_order(self):
        d: ConcurrentDict[int] = ConcurrentDict("test", 2)
        d.put("a", 1)
        d.put("b", 2)
        d.get("a")
        d.put("c", 3)
        assert d.get("a") == 1 and d.get("b") is None and d.get("c") == 3

    def test_remove(self):
        d: ConcurrentDict[str] = ConcurrentDict("test", 5)
        d.put("x", "y")
        assert d.remove("x") is True
        assert d.get("x") is None
        assert d.remove("x") is False

    def test_remove_prefix(self):
        d: ConcurrentDict[int] = ConcurrentDict("test", 10)
        d.put("/models/fe/m1#CPU", 1)
        d.put("/models/fe/m1#CUDA", 2)
        d.put("/models/fe/m2#CPU", 3)
        assert d.remove_prefix("/models/fe/m1") == 2
        assert d.get("/models/fe/m2#CPU") == 3

    def test_clear(self):
        d: ConcurrentDict[int] = ConcurrentDict("test", 5)
        d.put("a", 1)
        d.put("b", 2)
        assert d.clear() == 2 and d.size() == 0

    def test_concurrent_puts(self):
        d: ConcurrentDict[int] = ConcurrentDict("test", 100)
        errors = []

        def writer(n):
            try:
                for i in range(10):
                    d.put(f"key_{n}_{i}", n * 10 + i)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors

    def test_stats(self):
        d: ConcurrentDict[str] = ConcurrentDict("mydict", 5)
        d.put("a", "1")
        d.get("a")
        d.get("missing")
        s = d.stats()
        assert s["hits"] == 1 and s["misses"] == 1 and s["name"] == "mydict"
