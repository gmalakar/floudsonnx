# =============================================================================
# File: test_ranker.py
# Date Created: 2026-06-10
# Date Updated: 2026-06-12
# Copyright (c) 2026 Goutam Malakar.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================
"""Reranker / cross-encoder smoke test — sequence-classification."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))  # noqa: E402

import numpy as np  # noqa: E402

from floudsonnx import FloudsOnnxClient, FloudsOnnxSettings  # noqa: E402

MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"
MODEL_FOR = "ranker"
TASK = "sequence-classification"


def section(t: str) -> None:
    print(f"\n{'='*60}\n  {t}\n{'='*60}")


def run() -> None:
    client = FloudsOnnxClient(FloudsOnnxSettings())

    section("1. pull()")
    m = client.pull(MODEL, model_for=MODEL_FOR, task=TASK, library="transformers", optimize=True)
    print(f"  strategy  : {m.session_strategy}")
    print(f"  onnx files: {m.onnx_files}")

    section("2. create_model()")
    model = client.create_model(MODEL, model_for=MODEL_FOR)
    print(f"  {model}")
    assert model.session is not None
    inputs = [i.name for i in model.session.get_inputs()]
    outputs = [o.name for o in model.session.get_outputs()]
    print(f"  session inputs : {inputs}")
    print(f"  session outputs: {outputs}")

    section("3. rerank()")
    query = "What is machine learning?"
    passages = [
        "Machine learning is a type of artificial intelligence.",
        "The Eiffel Tower is located in Paris, France.",
        "Deep learning is a subset of machine learning using neural networks.",
        "Python is a popular programming language.",
    ]
    scores = []
    for passage in passages:
        enc = model.tokenizer(query, passage, return_tensors="np", padding=True, truncation=True, max_length=512)
        feed = {n: enc[n].astype(np.int64) for n in inputs if n in enc}
        for n in set(inputs) - set(feed):
            feed[n] = np.zeros_like(next(iter(feed.values())))
        out = model.run(None, feed)
        score = float(out[0][0][0] if out[0].ndim > 1 else out[0][0])
        scores.append((score, passage))

    scores.sort(reverse=True)
    print(f"\n  Query: {query!r}")
    print("  Ranked passages:")
    for rank, (score, passage) in enumerate(scores, 1):
        print(f"    {rank}. [{score:+.4f}] {passage[:60]}")
    print("  PASS")


if __name__ == "__main__":
    run()
