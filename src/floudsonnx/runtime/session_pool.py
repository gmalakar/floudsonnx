# =============================================================================
# Copyright (c) 2026 Goutam Malakar. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# =============================================================================
"""
floudsonnx.runtime.session_pool
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Thread-safe LRU caches for loaded ONNX Runtime sessions.

Three separate caches:
  _encoder_cache  — ort.InferenceSession      (fe, sc, ranker, decoder-only llm)
  _decoder_cache  — ort.InferenceSession      (decoder half of split s2s, future use)
  _seq2seq_cache  — ORTModelForSeq2SeqLM      (s2s, seq2seq llm)

Cache keys:
  encoder/decoder : "<abs_model_dir>#<provider>"
  seq2seq         : "<abs_model_dir>#seq2seq"

Double-checked locking per key prevents duplicate loads under concurrency.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Any

import onnxruntime as ort

from floudsonnx.config.model_config import ModelConfig
from floudsonnx.runtime.ort_seq2seq import OrtSeq2SeqLoader
from floudsonnx.runtime.ort_session import OrtInferenceSessionLoader
from floudsonnx.utils.concurrent_dict import ConcurrentDict

log = logging.getLogger(__name__)


class SessionPool:

    def __init__(self, encoder_max: int = 5, decoder_max: int = 5, seq2seq_max: int = 3) -> None:
        self._encoder_cache: ConcurrentDict[ort.InferenceSession] = ConcurrentDict("encoders", encoder_max)
        self._decoder_cache: ConcurrentDict[ort.InferenceSession] = ConcurrentDict("decoders", decoder_max)
        self._seq2seq_cache: ConcurrentDict[Any] = ConcurrentDict("seq2seq", seq2seq_max)
        self._load_locks: dict[str, threading.Lock] = {}
        self._lock_registry = threading.Lock()

    def get_or_load_encoder(self, model_dir: str, config: ModelConfig, provider: str = "CPUExecutionProvider") -> ort.InferenceSession:
        key = self._enc_key(model_dir, provider)
        cached = self._encoder_cache.get(key)
        if cached is not None:
            log.debug("session_pool: encoder HIT — %s", key)
            return cached
        with self._load_lock(key):
            cached = self._encoder_cache.get(key)
            if cached is not None:
                return cached
            log.debug("session_pool: encoder MISS — loading %s", key)
            session = OrtInferenceSessionLoader.load(model_dir, config, provider)
            self._encoder_cache.put(key, session)
            return session

    def get_or_load_seq2seq(self, model_dir: str, config: ModelConfig) -> Any:
        key = self._seq2seq_key(model_dir)
        cached = self._seq2seq_cache.get(key)
        if cached is not None:
            log.debug("session_pool: seq2seq HIT — %s", key)
            return cached
        with self._load_lock(key):
            cached = self._seq2seq_cache.get(key)
            if cached is not None:
                return cached
            log.debug("session_pool: seq2seq MISS — loading %s", key)
            model = OrtSeq2SeqLoader.load(model_dir, config)
            self._seq2seq_cache.put(key, model)
            return model

    def evict(self, model_dir: str) -> int:
        """Evict all cached sessions for *model_dir*. Returns count removed."""
        abs_dir = self._norm(model_dir)
        total = 0
        total += self._encoder_cache.remove_prefix(abs_dir)
        total += self._decoder_cache.remove_prefix(abs_dir)
        total += self._seq2seq_cache.remove_prefix(abs_dir)
        if total:
            log.info("session_pool: evicted %d session(s) for '%s'", total, abs_dir)
        return total

    def evict_all(self) -> int:
        total = self._encoder_cache.clear() + self._decoder_cache.clear() + self._seq2seq_cache.clear()
        log.info("session_pool: evicted all sessions (%d total)", total)
        return total

    def stats(self) -> dict:
        return {
            "encoder": self._encoder_cache.stats(),
            "decoder": self._decoder_cache.stats(),
            "seq2seq": self._seq2seq_cache.stats(),
        }

    def _load_lock(self, key: str) -> threading.Lock:
        with self._lock_registry:
            if key not in self._load_locks:
                self._load_locks[key] = threading.Lock()
            return self._load_locks[key]

    @staticmethod
    def _norm(path: str) -> str:
        return os.path.realpath(os.path.abspath(path))

    @classmethod
    def _enc_key(cls, model_dir: str, provider: str) -> str:
        return f"{cls._norm(model_dir)}#{provider}"

    @classmethod
    def _seq2seq_key(cls, model_dir: str) -> str:
        return f"{cls._norm(model_dir)}#seq2seq"
