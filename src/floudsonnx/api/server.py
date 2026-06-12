# =============================================================================
# Copyright (c) 2026 Goutam Malakar. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# =============================================================================
from __future__ import annotations

from typing import Any

from floudsonnx.exceptions import ServerNotInstalledError


def create_app() -> Any:
    try:
        from fastapi import FastAPI
    except ImportError as exc:
        raise ServerNotInstalledError() from exc
    from floudsonnx.api.routers import health, models

    app = FastAPI(title="floudsonnx", description="ONNX model store and runtime REST API", version="0.1.0")
    app.include_router(health.router)
    app.include_router(models.router, prefix="/api/v1")
    return app


def run_server(host: str = "127.0.0.1", port: int = 19720, reload: bool = False) -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise ServerNotInstalledError() from exc
    uvicorn.run("floudsonnx.api.server:create_app", factory=True, host=host, port=port, reload=reload)
