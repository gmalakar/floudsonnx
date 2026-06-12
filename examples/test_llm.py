# =============================================================================
# File: test_llm.py
# Date Created: 2026-06-10
# Date Updated: 2026-06-12
# Copyright (c) 2026 Goutam Malakar.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================
"""LLM smoke test — text-generation-with-past, TinyLlama chat."""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))  # noqa: E402

from floudsonnx import FloudsOnnxClient, FloudsOnnxSettings  # noqa: E402
from floudsonnx.config.model_config import ModelConfig  # noqa: E402

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MODEL_FOR = "llm"
TASK = "text-generation-with-past"


def section(t: str) -> None:
    print(f"\n{'='*60}\n  {t}\n{'='*60}")


def run() -> None:
    client = FloudsOnnxClient(FloudsOnnxSettings())

    section("1. pull()")
    config = ModelConfig(
        model_name=MODEL,
        model_for=MODEL_FOR,
        use_seq2seqlm=True,
        chat_template="tinyllama",
        model_family="tinyllama",
    )
    m = client.pull(MODEL, model_for=MODEL_FOR, task=TASK, library="transformers", opset_version=18, merge=True, config=config)
    print(f"  strategy  : {m.session_strategy}")
    print(f"  onnx files: {m.onnx_files}")

    section("2. create_model()")
    model = client.create_model(MODEL, model_for=MODEL_FOR)
    print(f"  {model}")
    print(f"  is_seq2seq : {model.is_seq2seq}")

    section("3. generate()")
    assert model.seq2seq_model is not None
    prompt = "What is the capital of France? Answer in one word."
    messages = [{"role": "user", "content": prompt}]
    try:
        text = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        text = prompt
    enc = model.tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    t0 = time.perf_counter()
    out_ids = model.seq2seq_model.generate(input_ids=enc["input_ids"], max_new_tokens=32, do_sample=False)
    elapsed = time.perf_counter() - t0
    new_tokens = out_ids[0][enc["input_ids"].shape[-1] :]
    decoded = model.tokenizer.decode(new_tokens, skip_special_tokens=True)
    print(f"  prompt  : {prompt!r}")
    print(f"  output  : {decoded!r}")
    print(f"  elapsed : {elapsed:.3f}s")
    print("  PASS")


if __name__ == "__main__":
    run()
