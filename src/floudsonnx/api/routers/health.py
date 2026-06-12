# =============================================================================
# Copyright (c) 2026 Goutam Malakar. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# =============================================================================
from __future__ import annotations

from typing import Any

try:
    from fastapi import APIRouter

    router: Any = APIRouter(tags=["health"])

    @router.get("/health")
    async def health():
        return {"status": "ok", "service": "floudsonnx"}

except ImportError:
    router = None
