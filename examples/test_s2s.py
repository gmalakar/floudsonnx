# =============================================================================
# File: test_s2s.py
# Date Created: 2026-06-10
# Date Updated: 2026-06-12
# Copyright (c) 2026 Goutam Malakar.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================
"""S2S model smoke test — seq2seq-lm, translate and summarize."""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))  # noqa: E402

from floudsonnx import FloudsOnnxClient, FloudsOnnxSettings  # noqa: E402

MODEL = "google/flan-t5-small"
MODEL_FOR = "s2s"
TASK = "seq2seq-lm"


def section(t: str) -> None:
    print(f"\n{'='*60}\n  {t}\n{'='*60}")


def run() -> None:
    client = FloudsOnnxClient(FloudsOnnxSettings())

    section("1. pull()")
    m = client.pull(MODEL, model_for=MODEL_FOR, task=TASK, library="transformers")
    print(f"  strategy  : {m.session_strategy}")
    print(f"  onnx files: {m.onnx_files}")

    section("2. create_model()")
    model = client.create_model(MODEL, model_for=MODEL_FOR)
    print(f"  {model}")
    print(f"  is_seq2seq : {model.is_seq2seq}")
    print(f"  tokenizer  : {type(model.tokenizer).__name__}")

    section("3. generate()")
    assert model.seq2seq_model is not None
    prompt = "Translate to French: The weather is nice today."
    enc = model.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
    t0 = time.perf_counter()
    out_ids = model.seq2seq_model.generate(input_ids=enc["input_ids"], max_new_tokens=64, num_beams=2)
    elapsed = time.perf_counter() - t0
    decoded = model.tokenizer.decode(out_ids[0], skip_special_tokens=True)
    print(f"  prompt  : {prompt!r}")
    print(f"  output  : {decoded!r}")
    print(f"  elapsed : {elapsed:.3f}s")
    print("  PASS")

    section("4. summarize()")
    text = (
        "summarize: The Amazon rainforest is a moist broadleaf tropical rainforest "
        "in the Amazon biome that covers most of the Amazon basin of South America."
    )
    enc = model.tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    t0 = time.perf_counter()
    out_ids = model.seq2seq_model.generate(enc["input_ids"], max_new_tokens=64)
    elapsed = time.perf_counter() - t0
    print(f"  summary : {model.tokenizer.decode(out_ids[0], skip_special_tokens=True)!r}")
    print(f"  elapsed : {elapsed:.3f}s")
    print("  PASS")


if __name__ == "__main__":
    run()
