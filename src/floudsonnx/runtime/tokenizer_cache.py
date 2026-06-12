# =============================================================================
# Copyright (c) 2026 Goutam Malakar. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# =============================================================================
"""
floudsonnx.runtime.tokenizer_cache
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Thread-local tokenizer cache.

Mirrors BaseNLPService._get_tokenizer_threadsafe():
  - Stored in threading.local() — never shared across threads.
  - AutoTokenizer.from_pretrained(path, local_files_only=True)
  - On failure → retry with use_fast=False (legacy mode).
  - Falls back to HuggingFace Hub for known broken local tokenizers.

Cache key: "<abs_model_dir>#legacy=<0|1>"
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict

from floudsonnx.exceptions import TokenizerError

log = logging.getLogger(__name__)

# Models whose local tokenizer is known to fail and must be pulled from the Hub
_HF_FALLBACKS: Dict[str, str] = {
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-t5-base": "sentence-transformers/sentence-t5-base",
}


class TokenizerCache:

    def __init__(self) -> None:
        self._local = threading.local()

    def get_or_load(self, model_dir: str, use_legacy: bool = False) -> Any:
        key = self._key(model_dir, use_legacy)
        store = self._store()
        if key in store:
            return store[key]
        tokenizer = self._load(model_dir, use_legacy)
        store[key] = tokenizer
        return tokenizer

    def evict(self, model_dir: str) -> None:
        abs_dir = os.path.realpath(os.path.abspath(model_dir))
        store = self._store()
        for k in [k for k in list(store.keys()) if k.startswith(abs_dir)]:
            del store[k]

    def clear(self) -> None:
        if hasattr(self._local, "tokenizers"):
            self._local.tokenizers.clear()

    def _store(self) -> dict:
        if not hasattr(self._local, "tokenizers"):
            self._local.tokenizers = {}
        return self._local.tokenizers

    @staticmethod
    def _key(model_dir: str, use_legacy: bool) -> str:
        return f"{os.path.realpath(os.path.abspath(model_dir))}#legacy={int(use_legacy)}"

    @staticmethod
    def _load(model_dir: str, use_legacy: bool) -> Any:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise TokenizerError("transformers is required for tokenizer loading.") from exc

        # Attempt 1: local, requested mode
        try:
            log.debug("TokenizerCache: loading from '%s' (legacy=%s)", model_dir, use_legacy)
            tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, use_fast=not use_legacy)
            log.info("TokenizerCache: ready — %s", os.path.basename(model_dir))
            return tok
        except Exception as e1:
            log.warning("TokenizerCache: attempt 1 failed (%s); trying legacy", e1)

        # Attempt 2: legacy fallback
        if not use_legacy:
            try:
                tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, use_fast=False)
                log.info("TokenizerCache: ready (legacy) — %s", os.path.basename(model_dir))
                return tok
            except Exception as e2:
                log.warning("TokenizerCache: attempt 2 failed (%s); trying HF Hub fallback", e2)

        # Attempt 3: known HF Hub fallback
        folder_name = os.path.basename(os.path.normpath(model_dir))
        hf_name = _HF_FALLBACKS.get(folder_name)
        if hf_name:
            try:
                log.info("TokenizerCache: downloading fallback '%s' from Hub", hf_name)
                tok = AutoTokenizer.from_pretrained(hf_name, use_fast=False)
                return tok
            except Exception as e3:
                log.error("TokenizerCache: HF Hub fallback failed for '%s': %s", hf_name, e3)

        raise TokenizerError(
            f"Could not load tokenizer from '{model_dir}'",
            detail="All attempts (local fast, local legacy, HF fallback) failed.",
        )
