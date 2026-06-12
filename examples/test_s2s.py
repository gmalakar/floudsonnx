# ==============================================================================
# examples/test_s2s.py
# S2S model smoke test — mirrors:
#   flouds-export export --model-for s2s \
#       --model-name google/flan-t5-small \
#       --task seq2seq-lm --library transformers
#
# Requires: pip install floudsonnx[export,seq2seq]
# ==============================================================================
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from floudsonnx import FloudsOnnxClient, FloudsOnnxSettings

MODEL = "google/flan-t5-small"   # small, fast to download and export
MODEL_FOR = "s2s"
TASK = "seq2seq-lm"


def section(t): print(f"\n{'='*60}\n  {t}\n{'='*60}")


def run():
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
    prompt = "Translate to French: The weather is nice today."
    enc = model.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
    t0 = time.perf_counter()
    out_ids = model.seq2seq_model.generate(
        input_ids=enc["input_ids"],
        max_new_tokens=64,
        num_beams=2,
    )
    elapsed = time.perf_counter() - t0
    decoded = model.tokenizer.decode(out_ids[0], skip_special_tokens=True)
    print(f"  prompt    : {prompt!r}")
    print(f"  output    : {decoded!r}")
    print(f"  elapsed   : {elapsed:.3f}s")
    print("  PASS")

    section("4. summarize()")
    text = (
        "summarize: The Amazon rainforest is a moist broadleaf tropical rainforest "
        "in the Amazon biome that covers most of the Amazon basin of South America. "
        "This basin encompasses 7,000,000 km2, of which 5,500,000 km2 are covered by "
        "the rainforest. This region includes territory belonging to nine nations."
    )
    enc = model.tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    t0 = time.perf_counter()
    out_ids = model.seq2seq_model.generate(enc["input_ids"], max_new_tokens=64)
    elapsed = time.perf_counter() - t0
    print(f"  summary   : {model.tokenizer.decode(out_ids[0], skip_special_tokens=True)!r}")
    print(f"  elapsed   : {elapsed:.3f}s")
    print("  PASS")


if __name__ == "__main__":
    run()
