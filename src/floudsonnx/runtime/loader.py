# =============================================================================
# Copyright (c) 2026 Goutam Malakar. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# =============================================================================
"""
floudsonnx.runtime.loader
~~~~~~~~~~~~~~~~~~~~~~~~~~
ModelLoader — main orchestrator.
LoadedModel — the object returned to all callers.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, List, Optional

import onnxruntime as ort

from floudsonnx.config.model_config import ModelConfig
from floudsonnx.config.settings import FloudsOnnxSettings
from floudsonnx.exceptions import ModelNotFoundError
from floudsonnx.runtime.session_pool import SessionPool
from floudsonnx.runtime.strategy import SessionStrategy, resolve_strategy
from floudsonnx.runtime.tokenizer_cache import TokenizerCache
from floudsonnx.store.manifest import ModelManifest
from floudsonnx.store.registry import ModelRegistry

log = logging.getLogger(__name__)


@dataclass
class LoadedModel:
    """
    Returned by create_model() / load_model().

    ORT_INFERENCE_SESSION:  session is set, seq2seq_model is None.
    ORT_SEQ2SEQ_LM:         seq2seq_model is set, session is None.

    model.run(output_names, input_feed)   — encoder models only.
    model.seq2seq_model.generate(...)     — seq2seq models.
    """

    model_name: str
    model_for: str
    model_dir: str
    config: ModelConfig
    tokenizer: Any
    session_strategy: SessionStrategy
    session: Optional[ort.InferenceSession] = field(default=None)
    seq2seq_model: Optional[Any] = field(default=None)

    @property
    def is_seq2seq(self) -> bool:
        return self.session_strategy == SessionStrategy.ORT_SEQ2SEQ_LM

    def run(self, output_names: Any, input_feed: Any, run_options: Any = None) -> Any:
        """Proxy to session.run(). For seq2seq use model.seq2seq_model.generate() instead."""
        if self.session is not None:
            return self.session.run(output_names, input_feed, run_options)
        raise TypeError(
            f"'{self.model_name}' uses ORTModelForSeq2SeqLM (strategy={self.session_strategy}). "
            "Use model.seq2seq_model.generate(...) for generation."
        )

    def __repr__(self) -> str:
        return (
            f"LoadedModel(model_name={self.model_name!r}, model_for={self.model_for!r}, "
            f"strategy={self.session_strategy.value}, "
            f"session={'ready' if self.session else 'n/a'}, "
            f"seq2seq={'ready' if self.seq2seq_model else 'n/a'})"
        )


class ModelLoader:
    """Orchestrates ModelRegistry + SessionPool + TokenizerCache."""

    def __init__(self, settings: Optional[FloudsOnnxSettings] = None) -> None:
        self._settings = settings or FloudsOnnxSettings()
        self._registry = ModelRegistry(self._settings)
        self._pool = SessionPool(
            encoder_max=self._settings.encoder_cache_max,
            decoder_max=self._settings.decoder_cache_max,
            seq2seq_max=self._settings.seq2seq_cache_max,
        )
        self._tokenizers = TokenizerCache()

    # ── Store verbs ───────────────────────────────────────────────────────────

    def pull(
        self,
        model_name: str,
        model_for: str = "fe",
        task: Optional[str] = None,
        force: bool = False,
        config: Optional[ModelConfig] = None,
        **kwargs: Any,
    ) -> ModelManifest:
        """Ensure artifact is on disk; export if missing. kwargs forwarded to ExporterBridge."""
        return self._registry.pull(
            model_name=model_name,
            model_for=model_for,
            task=task,
            force=force,
            config=config,
            **kwargs,
        )

    def list(self) -> List[ModelManifest]:
        return self._registry.list()

    def remove(self, model_name: str, model_for: str = "fe") -> bool:
        model_dir = self._registry.model_dir_path(model_name, model_for)
        self._pool.evict(model_dir)
        self._tokenizers.evict(model_dir)
        return self._registry.remove(model_name, model_for)

    # ── Runtime verbs ─────────────────────────────────────────────────────────

    def create_model(
        self,
        model_name: str,
        model_for: str = "fe",
        task: Optional[str] = None,
        force_pull: bool = False,
        config: Optional[ModelConfig] = None,
        **kwargs: Any,
    ) -> LoadedModel:
        """Pull (auto-export if missing) then load session. Idempotent."""
        manifest = self._registry.pull(
            model_name=model_name,
            model_for=model_for,
            task=task,
            force=force_pull,
            config=config,
            **kwargs,
        )
        return self._build(manifest)

    def load_model(self, model_name: str, model_for: str = "fe") -> LoadedModel:
        """Load a model already on disk. No auto-pull. Raises ModelNotFoundError if absent."""
        if not self._registry.exists(model_name, model_for):
            raise ModelNotFoundError(f"Model '{model_name}' (model_for={model_for}) not in local store. " "Use create_model() to auto-pull.")
        manifest = self._registry.get_manifest(model_name, model_for)
        if manifest is None:
            raise ModelNotFoundError(f"Manifest for '{model_name}' (model_for={model_for}) could not be read.")
        return self._build(manifest)

    def reload(self, model_name: str, model_for: str = "fe") -> LoadedModel:
        """Evict session from RAM and re-load from disk (hot-reload)."""
        if not self._registry.exists(model_name, model_for):
            raise ModelNotFoundError(f"Cannot reload '{model_name}' — not in local store.")
        model_dir = self._registry.model_dir_path(model_name, model_for)
        self._pool.evict(model_dir)
        self._tokenizers.evict(model_dir)
        manifest = self._registry.get_manifest(model_name, model_for)
        if manifest is None:
            raise ModelNotFoundError(f"Manifest for '{model_name}' (model_for={model_for}) could not be read.")
        log.info("reload: re-loading '%s/%s'", model_for, model_name)
        return self._build(manifest)

    def unload(self, model_name: str, model_for: str = "fe") -> bool:
        """Evict session from RAM; keep files on disk."""
        model_dir = self._registry.model_dir_path(model_name, model_for)
        evicted = self._pool.evict(model_dir)
        self._tokenizers.evict(model_dir)
        return evicted > 0

    def is_loaded(self, model_name: str, model_for: str = "fe") -> bool:
        model_dir = self._registry.model_dir_path(model_name, model_for)
        abs_dir = os.path.realpath(os.path.abspath(model_dir))
        for cache in (self._pool._encoder_cache, self._pool._decoder_cache, self._pool._seq2seq_cache):
            if any(k.startswith(abs_dir) for k in cache.keys()):
                return True
        return False

    def cache_stats(self) -> dict[str, Any]:
        return self._pool.stats()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _build(self, manifest: ModelManifest) -> LoadedModel:
        config = manifest.config or ModelConfig(model_name=manifest.model_name, model_for=manifest.model_for)
        strategy = resolve_strategy(config)
        model_dir = self._registry.model_dir_path(manifest.model_name, manifest.model_for)
        provider = self._settings.session_provider
        tokenizer = self._tokenizers.get_or_load(model_dir, use_legacy=config.legacy_tokenizer)

        if strategy == SessionStrategy.ORT_INFERENCE_SESSION:
            session = self._pool.get_or_load_encoder(model_dir, config, provider)
            return LoadedModel(
                model_name=manifest.model_name,
                model_for=manifest.model_for,
                model_dir=model_dir,
                config=config,
                tokenizer=tokenizer,
                session_strategy=strategy,
                session=session,
                seq2seq_model=None,
            )
        else:
            seq2seq = self._pool.get_or_load_seq2seq(model_dir, config)
            return LoadedModel(
                model_name=manifest.model_name,
                model_for=manifest.model_for,
                model_dir=model_dir,
                config=config,
                tokenizer=tokenizer,
                session_strategy=strategy,
                session=None,
                seq2seq_model=seq2seq,
            )
