# =============================================================================
# Copyright (c) 2026 Goutam Malakar. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# =============================================================================
from __future__ import annotations

from typing import Annotated, Any, Optional

try:
    from fastapi import APIRouter, HTTPException, Query
    from pydantic import BaseModel as _BM

    from floudsonnx.api.client import get_default_client
    from floudsonnx.exceptions import FloudsOnnxError

    router = APIRouter(tags=["models"])

    class PullRequest(_BM):
        model_name: str
        model_for: str = "fe"
        task: Optional[str] = None
        force: bool = False
        optimize: Optional[bool] = None
        trust_remote_code: bool = False
        use_external_data_format: bool = False
        use_fallback_if_failed: bool = False
        hf_token: Optional[str] = None

    class LoadRequest(_BM):
        model_name: str
        model_for: str = "fe"

    @router.get("/models")
    async def list_models() -> list[dict[str, Any]]:
        return [m.model_dump(mode="json") for m in get_default_client().list()]

    @router.get("/models/{name:path}")
    async def get_model(name: str, model_for: Annotated[str, Query()] = "fe") -> dict[str, Any]:
        manifest = get_default_client()._loader._registry.get_manifest(name, model_for)
        if manifest is None:
            raise HTTPException(status_code=404, detail=f"Model '{name}' not found")
        return manifest.model_dump(mode="json")

    @router.post("/models/pull")
    async def pull_model(req: PullRequest) -> dict[str, Any]:
        try:
            manifest = get_default_client().pull(
                req.model_name,
                req.model_for,
                task=req.task,
                force=req.force,
                optimize=req.optimize,
                trust_remote_code=req.trust_remote_code,
                use_external_data_format=req.use_external_data_format,
                use_fallback_if_failed=req.use_fallback_if_failed,
                hf_token=req.hf_token,
            )
            return manifest.model_dump(mode="json")
        except FloudsOnnxError as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @router.post("/models/load")
    async def load_model(req: LoadRequest) -> dict[str, str]:
        try:
            loaded = get_default_client().load_model(req.model_name, req.model_for)
            return {"model_name": loaded.model_name, "model_for": loaded.model_for, "strategy": loaded.session_strategy.value, "status": "loaded"}
        except FloudsOnnxError as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @router.post("/models/reload")
    async def reload_model(req: LoadRequest) -> dict[str, str]:
        try:
            loaded = get_default_client().reload(req.model_name, req.model_for)
            return {"model_name": loaded.model_name, "strategy": loaded.session_strategy.value, "status": "reloaded"}
        except FloudsOnnxError as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @router.post("/models/unload")
    async def unload_model(req: LoadRequest) -> dict[str, Any]:
        evicted = get_default_client().unload(req.model_name, req.model_for)
        return {"model_name": req.model_name, "evicted": evicted}

    @router.delete("/models/{name:path}")
    async def remove_model(name: str, model_for: Annotated[str, Query()] = "fe") -> dict[str, Any]:
        removed = get_default_client().remove(name, model_for)
        return {"model_name": name, "removed": removed}

    @router.get("/stats")
    async def stats() -> dict[str, Any]:
        return get_default_client().cache_stats()

except ImportError:
    router = None
