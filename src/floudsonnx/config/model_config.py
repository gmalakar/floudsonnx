# =============================================================================
# Copyright (c) 2026 Goutam Malakar. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# =============================================================================
"""
floudsonnx.config.model_config
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Self-contained ModelConfig — clean replacement for OnnxConfig from model_service.
All fields mirror OnnxConfig for manifest compatibility.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class InputNames(BaseModel):
    input: str = "input_ids"
    mask: str = "attention_mask"
    position: Optional[str] = None
    token_type: Optional[str] = None
    decoder_input: str = "decoder_input_ids"


class OutputNames(BaseModel):
    output: str = "last_hidden_state"


class DecoderInputNames(BaseModel):
    input: str = "input_ids"
    mask: str = "encoder_attention_mask"
    encoder_output: str = "encoder_hidden_states"


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    # ── Identity ──────────────────────────────────────────────────────────────
    model_name: str
    model_for: str = "fe"  # fe | s2s | sc | ranker | llm
    tasks: List[str] = Field(default_factory=list)
    model_folder_name: Optional[str] = None

    # ── Encoder / embedding ───────────────────────────────────────────────────
    dimension: Optional[int] = None
    native_dimension: Optional[int] = None
    max_length: int = 256
    min_length: int = 0
    pooling_strategy: str = "mean"
    normalize: bool = False
    force_pooling: bool = True
    lowercase: bool = True
    remove_emojis: bool = False
    legacy_tokenizer: bool = False
    chunk_logic: str = "sentence"
    chunk_overlap: int = 1
    chunk_size: Optional[int] = None

    # ── ONNX artifact names ───────────────────────────────────────────────────
    encoder_onnx_model: str = "model.onnx"
    optimized_onnx_model: str = "model_optimized.onnx"
    decoder_onnx_model: str = "decoder_model.onnx"
    decoder_onnx_model_with_past: Optional[str] = None

    # ── Architecture flags ────────────────────────────────────────────────────
    encoder_only: bool = False
    decoder_only: bool = False
    use_seq2seqlm: bool = False
    merged_with_past: bool = False
    use_cache: Optional[bool] = None

    # ── I/O name mappings ─────────────────────────────────────────────────────
    inputnames: InputNames = Field(default_factory=InputNames)
    outputnames: OutputNames = Field(default_factory=OutputNames)
    decoder_inputnames: DecoderInputNames = Field(default_factory=DecoderInputNames)

    # ── Generation parameters ─────────────────────────────────────────────────
    num_beams: int = 4
    temperature: float = 0.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
    early_stopping: bool = True
    max_new_tokens: Optional[int] = 512
    forced_bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    bos_token_id: Optional[int] = None
    pad_token_id: int = 0
    vocab_size: Optional[int] = None
    prepend_text: str = "summarize: "

    # ── Chat / LLM ────────────────────────────────────────────────────────────
    model_family: Optional[str] = None
    chat_template: Optional[str] = None
    extract_assistant_only: bool = False
    assistant_prefix: str = "assistant:"

    # ── Quantization ─────────────────────────────────────────────────────────
    quantize: bool = False
    quantize_type: str = "int8"

    # ── Special token file names ──────────────────────────────────────────────
    special_tokens_map_path: str = "special_tokens_map.json"
    generation_config_path: str = "generation_config.json"

    # ── Extra / pass-through ──────────────────────────────────────────────────
    extra: Dict[str, Any] = Field(default_factory=dict)
