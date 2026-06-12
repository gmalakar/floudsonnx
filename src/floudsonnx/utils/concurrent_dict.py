# =============================================================================
# Copyright (c) 2026 Goutam Malakar. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# =============================================================================
"""
Thread-safe LRU dictionary. Self-contained — no monorepo dependency.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Generic, TypeVar

V = TypeVar("V")


class ConcurrentDict(Generic[V]):
    """
    Thread-safe LRU cache with a fixed capacity.
    Least-recently-used entry is evicted when capacity is exceeded.
    """

    def __init__(self, name: str, capacity: int = 10) -> None:
        if capacity < 1:
            raise ValueError(f"ConcurrentDict '{name}' capacity must be >= 1, got {capacity}")
        self._name = name
        self._capacity = capacity
        self._store: OrderedDict[str, V] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> V | None:
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                self._hits += 1
                return self._store[key]
            self._misses += 1
            return None

    def put(self, key: str, value: V) -> None:
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                self._store[key] = value
            else:
                self._store[key] = value
                if len(self._store) > self._capacity:
                    self._store.popitem(last=False)

    def remove(self, key: str) -> bool:
        with self._lock:
            if key in self._store:
                del self._store[key]
                return True
            return False

    def remove_prefix(self, prefix: str) -> int:
        """Remove all entries whose key starts with *prefix*. Returns count removed."""
        with self._lock:
            keys = [k for k in self._store if k.startswith(prefix)]
            for k in keys:
                del self._store[k]
            return len(keys)

    def clear(self) -> int:
        with self._lock:
            count = len(self._store)
            self._store.clear()
            return count

    def size(self) -> int:
        with self._lock:
            return len(self._store)

    def keys(self) -> list[str]:
        with self._lock:
            return list(self._store.keys())

    def stats(self) -> dict:
        with self._lock:
            return {
                "name": self._name,
                "capacity": self._capacity,
                "size": len(self._store),
                "hits": self._hits,
                "misses": self._misses,
            }
