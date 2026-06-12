# =============================================================================
# Copyright (c) 2026 Goutam Malakar. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# =============================================================================
"""
Path safety helpers.
"""
from __future__ import annotations

import os
import re

_UNSAFE_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def safe_model_folder(model_name: str) -> str:
    """
    Return the last path segment of a HuggingFace model name as a safe folder name.

    >>> safe_model_folder("sentence-transformers/all-MiniLM-L6-v2")
    'all-MiniLM-L6-v2'
    >>> safe_model_folder("t5-small")
    't5-small'
    """
    folder = model_name.strip().rstrip("/").split("/")[-1]
    folder = _UNSAFE_CHARS.sub("_", folder)
    if not folder:
        raise ValueError(f"Cannot derive a safe folder name from model_name={model_name!r}")
    return folder


def safe_join(base: str, *parts: str) -> str:
    """
    Join *base* with *parts* and verify the result stays inside *base*.
    Raises ValueError on path-traversal attempts.
    """
    base_abs = os.path.realpath(os.path.abspath(base))
    joined = os.path.realpath(os.path.abspath(os.path.join(base, *parts)))
    if not joined.startswith(base_abs + os.sep) and joined != base_abs:
        raise ValueError(f"Path traversal detected: '{os.path.join(*parts)}' escapes base '{base}'")
    return joined
