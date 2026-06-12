# ==============================================================================
# examples/test_ranker.py
# Reranker / cross-encoder smoke test — mirrors:
#   flouds-export export --model-for ranker \
#       --model-name cross-encoder/ms-marco-MiniLM-L-12-v2 \
#       --task sequence-classification --library transformers --optimize
#
# A reranker scores (query, passage) pairs — output is a single logit per pair.
# ==============================================================================
from __future__ import annotations

import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from floudsonnx import FloudsOnnxClient, FloudsOnnxSettings

MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"   # small, fast
MODEL_FOR = "ranker"
TASK = "sequence-classification"


def section(t): print(f"\n{'='*60}\n  {t}\n{'='*60}")


def run():
    client = FloudsOnnxClient(FloudsOnnxSettings())

    section("1. pull()")
    m = client.pull(
        MODEL,
        model_for=MODEL_FOR,
        task=TASK,
        library="transformers",
        optimize=True,
    )
    print(f"  strategy  : {m.session_strategy}")
    print(f"  onnx files: {m.onnx_files}")

    section("2. create_model()")
    model = client.create_model(MODEL, model_for=MODEL_FOR)
    print(f"  {model}")
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
        enc = model.tokenizer(
            query, passage,
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=512,
        )
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
