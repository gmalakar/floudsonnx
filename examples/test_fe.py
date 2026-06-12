# ==============================================================================
# examples/test_fe.py
# FE model smoke test — mirrors:
#   flouds-export export --model-for fe \
#       --model-name sentence-transformers/all-MiniLM-L6-v2 \
#       --task feature-extraction --library transformers --normalize-embeddings
# ==============================================================================
from __future__ import annotations

import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from floudsonnx import FloudsOnnxClient, FloudsOnnxSettings

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_FOR = "fe"


def section(t): print(f"\n{'='*60}\n  {t}\n{'='*60}")


def run():
    client = FloudsOnnxClient(FloudsOnnxSettings())

    section("1. pull()")
    m = client.pull(MODEL, model_for=MODEL_FOR, library="transformers", normalize_embeddings=True)
    print(f"  strategy  : {m.session_strategy}")
    print(f"  onnx files: {m.onnx_files}")

    section("2. create_model()")
    model = client.create_model(MODEL, model_for=MODEL_FOR)
    print(f"  {model}")
    inputs = [i.name for i in model.session.get_inputs()]
    print(f"  session inputs : {inputs}")

    section("3. inference")
    texts = ["The quick brown fox.", "Hello world."]
    enc = model.tokenizer(texts, return_tensors="np", padding=True, truncation=True, max_length=256)
    feed = {n: enc[n].astype(np.int64) for n in inputs if n in enc}
    for n in set(inputs) - set(feed):
        feed[n] = np.zeros_like(next(iter(feed.values())))
    t0 = time.perf_counter()
    out = model.run(None, feed)
    print(f"  output shape: {out[0].shape}  elapsed: {time.perf_counter()-t0:.3f}s")
    print("  PASS")


if __name__ == "__main__":
    run()
