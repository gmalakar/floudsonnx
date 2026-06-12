# =============================================================================
# File: manual_test.py
# Date Created: 2026-06-10
# Date Updated: 2026-06-12
# Copyright (c) 2026 Goutam Malakar.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================
"""
Quick manual smoke test for floudsonnx.

Prerequisites:
    pip install floudsonnx[export]
    do NOT install sentence-transformers — use library=transformers

Usage:
    python manual_test.py
    python manual_test.py --model t5-small --for s2s
    python manual_test.py --no-export
    python manual_test.py --home C:/tmp/flouds
"""
from __future__ import annotations

import argparse
import json
import sys
import time


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def run(args: argparse.Namespace) -> int:  # noqa: C901
    from floudsonnx import FloudsOnnxClient, FloudsOnnxSettings

    settings = FloudsOnnxSettings(
        home_dir=args.home,
        export_optimize=args.optimize,
        session_provider="CPUExecutionProvider",
    )
    client = FloudsOnnxClient(settings)
    model_name = args.model
    model_for = args.model_for

    library = args.library
    if library is None and model_for in ("fe", "sc", "ranker"):
        library = "transformers"

    print("\nfloudsonnx manual test")
    print(f"  model               : {model_name}")
    print(f"  model_for           : {model_for}")
    print(f"  task                : {args.task or '(auto)'}")
    print(f"  library             : {library or 'auto'}")
    print(f"  normalize_embeddings: {args.normalize_embeddings}")
    print(f"  optimize            : {args.optimize}")
    print(f"  store               : {settings.home_dir}")

    # ── 1. PULL ───────────────────────────────────────────────────────────────
    if not args.no_export:
        section("1. pull()")
        t0 = time.perf_counter()
        manifest = client.pull(
            model_name,
            model_for=model_for,
            task=args.task or None,
            library=library,
            normalize_embeddings=args.normalize_embeddings,
        )
        elapsed = time.perf_counter() - t0
        print(f"  pulled_at  : {manifest.pulled_at}")
        print(f"  strategy   : {manifest.session_strategy}")
        print(f"  onnx files : {manifest.onnx_files}")
        print(f"  elapsed    : {elapsed:.1f}s")
    else:
        section("1. pull() — SKIPPED (--no-export)")

    # ── 2. LIST ───────────────────────────────────────────────────────────────
    section("2. list()")
    manifests = client.list()
    print(f"  models in store: {len(manifests)}")
    for m in manifests:
        print(f"    {m.model_for:8}  {m.model_name}")

    # ── 3. CREATE MODEL ───────────────────────────────────────────────────────
    section("3. create_model()")
    t0 = time.perf_counter()
    model = client.create_model(model_name, model_for=model_for)
    elapsed = time.perf_counter() - t0
    print(f"  {model}")
    print(f"  elapsed      : {elapsed:.3f}s")
    print(f"  is_seq2seq   : {model.is_seq2seq}")
    print(f"  tokenizer    : {type(model.tokenizer).__name__}")
    if not model.is_seq2seq and model.session is not None:
        print(f"  session inputs : {[i.name for i in model.session.get_inputs()]}")
        print(f"  session outputs: {[o.name for o in model.session.get_outputs()]}")

    # ── 4. INFERENCE ──────────────────────────────────────────────────────────
    section("4. inference")
    texts = ["The quick brown fox jumps over the lazy dog.", "Hello world."]

    if not model.is_seq2seq:
        import numpy as np

        assert model.session is not None, "Expected InferenceSession for non-seq2seq model"
        enc = model.tokenizer(texts, return_tensors="np", padding=True, truncation=True, max_length=model.config.max_length)
        session_inputs = {inp.name for inp in model.session.get_inputs()}
        input_feed = {name: enc[name].astype(np.int64) for name in session_inputs if name in enc}
        missing = session_inputs - set(input_feed.keys())
        if missing:
            print(f"  WARN: tokenizer did not produce {missing}; adding zeros")
            for name in missing:
                ref = next(iter(input_feed.values()))
                input_feed[name] = np.zeros_like(ref)
        t0 = time.perf_counter()
        outputs = model.run(None, input_feed)
        elapsed = time.perf_counter() - t0
        print(f"  input names    : {list(input_feed.keys())}")
        print(f"  output count   : {len(outputs)}")
        print(f"  output[0] shape: {outputs[0].shape}")
        print(f"  elapsed        : {elapsed:.3f}s")
        print("  PASS: session.run() returned output")
    else:
        assert model.seq2seq_model is not None, "Expected seq2seq_model for seq2seq model"
        enc = model.tokenizer(
            [f"{model.config.prepend_text}{texts[0]}"],
            return_tensors="pt",
            truncation=True,
            max_length=model.config.max_length,
        )
        t0 = time.perf_counter()
        out_ids = model.seq2seq_model.generate(input_ids=enc["input_ids"], max_new_tokens=64)
        elapsed = time.perf_counter() - t0
        decoded = model.tokenizer.decode(out_ids[0], skip_special_tokens=True)
        print(f"  generated      : {decoded!r}")
        print(f"  elapsed        : {elapsed:.3f}s")
        print("  PASS: seq2seq_model.generate() returned output")

    # ── 5. CACHE HIT ─────────────────────────────────────────────────────────
    section("5. cache hit (second create_model call)")
    t0 = time.perf_counter()
    model2 = client.create_model(model_name, model_for=model_for)
    elapsed = time.perf_counter() - t0
    same = (model2.session is model.session) if not model.is_seq2seq else (model2.seq2seq_model is model.seq2seq_model)
    print(f"  elapsed      : {elapsed:.4f}s  (should be <0.01s)")
    print(f"  same session : {same}")
    print("  PASS" if same else "  WARN: session was not reused")

    # ── 6. IS_LOADED ─────────────────────────────────────────────────────────
    section("6. is_loaded()")
    loaded = client.is_loaded(model_name, model_for=model_for)
    print(f"  is_loaded    : {loaded}")
    print("  PASS" if loaded else "  FAIL")

    # ── 7. CACHE STATS ────────────────────────────────────────────────────────
    section("7. cache_stats()")
    print(json.dumps(client.cache_stats(), indent=2))

    # ── 8. RELOAD ─────────────────────────────────────────────────────────────
    section("8. reload()")
    t0 = time.perf_counter()
    reloaded = client.reload(model_name, model_for=model_for)
    elapsed = time.perf_counter() - t0
    print(f"  {reloaded}")
    print(f"  elapsed      : {elapsed:.3f}s")
    print("  PASS")

    # ── 9. UNLOAD ─────────────────────────────────────────────────────────────
    section("9. unload()")
    evicted = client.unload(model_name, model_for=model_for)
    print(f"  evicted      : {evicted}")
    print(f"  is_loaded    : {client.is_loaded(model_name, model_for=model_for)}")
    print("  PASS" if evicted else "  WARN: nothing evicted")

    section("ALL STEPS PASSED")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="floudsonnx manual smoke test")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--for", dest="model_for", default="fe")
    parser.add_argument("--task", default="", help="Export task (default: auto from model_for)")
    parser.add_argument("--library", default=None, help="transformers | sentence_transformers")
    parser.add_argument("--normalize-embeddings", action="store_true", default=False)
    parser.add_argument("--optimize", action="store_true", default=False)
    parser.add_argument("--home", default="~/.flouds")
    parser.add_argument("--no-export", action="store_true")
    args = parser.parse_args()

    from pathlib import Path

    args.home = Path(args.home).expanduser()

    try:
        sys.exit(run(args))
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)
    except Exception as exc:
        import traceback

        print(f"\nFAIL: {exc}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
