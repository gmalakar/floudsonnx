# ==============================================================================
# examples/test_llm.py
# LLM model smoke test — mirrors:
#   flouds-export export --model-for llm \
#       --model-name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
#       --task text-generation-with-past --library transformers \
#       --opset-version 18 --merge
#
# Using TinyLlama — smallest practical LLM (~1.1B params, ~2.2GB).
# Requires: pip install floudsonnx[export,seq2seq]
# Note: generation uses seq2seq_model path (use_seq2seqlm=True in config).
# ==============================================================================
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from floudsonnx import FloudsOnnxClient, FloudsOnnxSettings
from floudsonnx.config.model_config import ModelConfig

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MODEL_FOR = "llm"
TASK = "text-generation-with-past"


def section(t): print(f"\n{'='*60}\n  {t}\n{'='*60}")


def run():
    client = FloudsOnnxClient(FloudsOnnxSettings())

    section("1. pull()")
    # use_seq2seqlm=True so floudsonnx loads via ORTModelForSeq2SeqLM
    config = ModelConfig(
        model_name=MODEL,
        model_for=MODEL_FOR,
        use_seq2seqlm=True,
        chat_template="tinyllama",
        model_family="tinyllama",
    )
    m = client.pull(
        MODEL,
        model_for=MODEL_FOR,
        task=TASK,
        library="transformers",
        opset_version=18,
        merge=True,
        config=config,
    )
    print(f"  strategy  : {m.session_strategy}")
    print(f"  onnx files: {m.onnx_files}")

    section("2. create_model()")
    model = client.create_model(MODEL, model_for=MODEL_FOR)
    print(f"  {model}")
    print(f"  is_seq2seq : {model.is_seq2seq}")

    section("3. generate()")
    prompt = "What is the capital of France? Answer in one word."
    # TinyLlama uses a chat template
    messages = [{"role": "user", "content": prompt}]
    try:
        text = model.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        text = prompt
    enc = model.tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    t0 = time.perf_counter()
    out_ids = model.seq2seq_model.generate(
        input_ids=enc["input_ids"],
        max_new_tokens=32,
        do_sample=False,
    )
    elapsed = time.perf_counter() - t0
    # Decode only generated tokens (skip input)
    new_tokens = out_ids[0][enc["input_ids"].shape[-1]:]
    decoded = model.tokenizer.decode(new_tokens, skip_special_tokens=True)
    print(f"  prompt  : {prompt!r}")
    print(f"  output  : {decoded!r}")
    print(f"  elapsed : {elapsed:.3f}s")
    print("  PASS")


if __name__ == "__main__":
    run()
