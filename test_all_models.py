# ==============================================================================
# test_all_models.py
# Unified manual smoke test for all model types: fe, s2s, ranker, llm.
#
# Prerequisites:
#   pip install floudsonnx[export,seq2seq]
#   do NOT install sentence-transformers (use --library transformers)
#
# Default models (small/fast):
#   fe     : sentence-transformers/all-MiniLM-L6-v2
#   s2s    : google/flan-t5-small
#   ranker : cross-encoder/ms-marco-MiniLM-L-12-v2
#   llm    : TinyLlama/TinyLlama-1.1B-Chat-v1.0
#
# Usage:
#   python test_all_models.py                         # all types
#   python test_all_models.py --type fe               # single type
#   python test_all_models.py --type fe s2s ranker    # multiple types
#   python test_all_models.py --no-export             # skip pull (load existing)
#   python test_all_models.py --home C:/tmp/flouds
#   python test_all_models.py --type llm --no-export  # load existing LLM
# ==============================================================================
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Model catalogue — mirrors the batch export commands exactly
# ---------------------------------------------------------------------------

@dataclass
class ModelSpec:
    model_for: str
    model_name: str
    task: str
    library: str = "transformers"
    normalize_embeddings: bool = False
    optimize: bool = False
    opset_version: Optional[int] = None
    merge: bool = False
    trust_remote_code: bool = False
    use_seq2seqlm: bool = False          # for LLMs loaded via ORTModelForSeq2SeqLM
    chat_template: Optional[str] = None
    extra_pull: dict = field(default_factory=dict)
    # Inference prompts
    texts: list = field(default_factory=lambda: [
        "The quick brown fox jumps over the lazy dog.",
        "Hello world.",
    ])
    gen_prompt: str = "Summarize: The Amazon rainforest covers most of the Amazon basin."
    max_new_tokens: int = 64


CATALOGUE: dict[str, ModelSpec] = {
    "fe": ModelSpec(
        model_for="fe",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        task="feature-extraction",
        library="transformers",
        normalize_embeddings=True,
    ),
    "s2s": ModelSpec(
        model_for="s2s",
        model_name="google/flan-t5-small",
        task="seq2seq-lm",
        library="transformers",
        use_seq2seqlm=True,
        gen_prompt="Translate to French: The weather is nice today.",
    ),
    "ranker": ModelSpec(
        model_for="ranker",
        model_name="cross-encoder/ms-marco-MiniLM-L-12-v2",
        task="sequence-classification",
        library="transformers",
        optimize=True,
        texts=[
            "What is machine learning?",
            "Machine learning is a type of artificial intelligence.",
            "The Eiffel Tower is in Paris.",
            "Deep learning is a subset of machine learning.",
        ],
    ),
    "llm": ModelSpec(
        model_for="llm",
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        task="text-generation-with-past",
        library="transformers",
        opset_version=18,
        merge=True,
        use_seq2seqlm=True,
        chat_template="tinyllama",
        gen_prompt="What is the capital of France? Answer in one word.",
        max_new_tokens=32,
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def section(title: str) -> None:
    print(f"\n{'=' * 62}")
    print(f"  {title}")
    print(f"{'=' * 62}")


def ok(msg: str = "") -> None:
    print(f"  ✓  PASS{': ' + msg if msg else ''}")


def warn(msg: str) -> None:
    print(f"  ⚠  WARN: {msg}")


def _build_input_feed(model: Any, texts: list, numpy) -> dict:
    """Build session input feed, auto-filling missing keys with zeros."""
    enc = model.tokenizer(
        texts,
        return_tensors="np",
        padding=True,
        truncation=True,
        max_length=model.config.max_length,
    )
    session_inputs = {inp.name for inp in model.session.get_inputs()}
    feed = {n: enc[n].astype(numpy.int64) for n in session_inputs if n in enc}
    missing = session_inputs - set(feed)
    if missing:
        warn(f"tokenizer did not produce {missing}; filling with zeros")
        for n in missing:
            feed[n] = numpy.zeros_like(next(iter(feed.values())))
    return feed


# ---------------------------------------------------------------------------
# Per-type inference
# ---------------------------------------------------------------------------

def _infer_fe_sc(model: Any, spec: ModelSpec) -> None:
    import numpy as np
    feed = _build_input_feed(model, spec.texts, np)
    t0 = time.perf_counter()
    out = model.run(None, feed)
    elapsed = time.perf_counter() - t0
    print(f"  input names    : {list(feed.keys())}")
    print(f"  output count   : {len(out)}")
    print(f"  output[0] shape: {out[0].shape}")
    print(f"  elapsed        : {elapsed:.3f}s")
    ok("session.run() returned output")


def _infer_ranker(model: Any, spec: ModelSpec) -> None:
    """Score query vs passages — texts[0] is query, texts[1:] are passages."""
    import numpy as np
    query = spec.texts[0]
    passages = spec.texts[1:]
    inputs = [i.name for i in model.session.get_inputs()]
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
        score = float(out[0].flatten()[0])
        scores.append((score, passage))
    scores.sort(reverse=True)
    print(f"  query          : {query!r}")
    for rank, (score, passage) in enumerate(scores, 1):
        print(f"  rank {rank}: [{score:+.4f}] {passage[:55]}")
    ok("reranking returned scores")


def _infer_s2s(model: Any, spec: ModelSpec) -> None:
    """Generate text from seq2seq or LLM model."""
    prompt = spec.gen_prompt

    # LLM with chat template
    if spec.chat_template:
        messages = [{"role": "user", "content": prompt}]
        try:
            text = model.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            text = prompt
    else:
        text = f"{model.config.prepend_text}{prompt}"

    enc = model.tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    t0 = time.perf_counter()
    out_ids = model.seq2seq_model.generate(
        input_ids=enc["input_ids"],
        max_new_tokens=spec.max_new_tokens,
        do_sample=False,
    )
    elapsed = time.perf_counter() - t0

    # For LLMs decode only newly generated tokens
    if spec.chat_template:
        new_tokens = out_ids[0][enc["input_ids"].shape[-1]:]
        decoded = model.tokenizer.decode(new_tokens, skip_special_tokens=True)
    else:
        decoded = model.tokenizer.decode(out_ids[0], skip_special_tokens=True)

    print(f"  prompt  : {prompt!r}")
    print(f"  output  : {decoded!r}")
    print(f"  elapsed : {elapsed:.3f}s")
    ok("generate() returned output")


# ---------------------------------------------------------------------------
# Core 9-step test runner (mirrors manual_test.py exactly)
# ---------------------------------------------------------------------------

def run_model_test(spec: ModelSpec, client: Any, no_export: bool) -> bool:
    model_name = spec.model_name
    model_for = spec.model_for
    passed = True

    print(f"\n{'#' * 62}")
    print(f"#  MODEL TYPE : {model_for.upper()}")
    print(f"#  MODEL NAME : {model_name}")
    print(f"{'#' * 62}")
    print(f"  task                : {spec.task}")
    print(f"  library             : {spec.library}")
    print(f"  normalize_embeddings: {spec.normalize_embeddings}")
    print(f"  optimize            : {spec.optimize}")
    print(f"  use_seq2seqlm       : {spec.use_seq2seqlm}")
    if spec.opset_version:
        print(f"  opset_version       : {spec.opset_version}")
    if spec.merge:
        print(f"  merge               : {spec.merge}")

    try:
        # ── 1. PULL ───────────────────────────────────────────────────────────
        if not no_export:
            section("1. pull()")
            from floudsonnx.config.model_config import ModelConfig
            config = None
            if spec.use_seq2seqlm:
                config = ModelConfig(
                    model_name=model_name,
                    model_for=model_for,
                    use_seq2seqlm=True,
                    chat_template=spec.chat_template,
                )
            pull_kwargs = dict(
                model_for=model_for,
                task=spec.task,
                library=spec.library,
                normalize_embeddings=spec.normalize_embeddings,
                optimize=spec.optimize,
                trust_remote_code=spec.trust_remote_code,
                merge=spec.merge,
                config=config,
            )
            if spec.opset_version:
                pull_kwargs["opset_version"] = spec.opset_version
            pull_kwargs.update(spec.extra_pull)

            t0 = time.perf_counter()
            manifest = client.pull(model_name, **pull_kwargs)
            elapsed = time.perf_counter() - t0
            print(f"  pulled_at  : {manifest.pulled_at}")
            print(f"  strategy   : {manifest.session_strategy}")
            print(f"  onnx files : {manifest.onnx_files}")
            print(f"  elapsed    : {elapsed:.1f}s")
            ok()
        else:
            section("1. pull() — SKIPPED (--no-export)")

        # ── 2. LIST ───────────────────────────────────────────────────────────
        section("2. list()")
        manifests = client.list()
        print(f"  models in store: {len(manifests)}")
        for m in manifests:
            marker = " ◄" if m.model_name == model_name else ""
            print(f"    {m.model_for:8}  {m.model_name}{marker}")
        ok()

        # ── 3. CREATE MODEL ───────────────────────────────────────────────────
        section("3. create_model()")
        t0 = time.perf_counter()
        model = client.create_model(model_name, model_for=model_for)
        elapsed = time.perf_counter() - t0
        print(f"  {model}")
        print(f"  elapsed      : {elapsed:.3f}s")
        print(f"  is_seq2seq   : {model.is_seq2seq}")
        print(f"  tokenizer    : {type(model.tokenizer).__name__}")
        if not model.is_seq2seq:
            print(f"  session in   : {[i.name for i in model.session.get_inputs()]}")
            print(f"  session out  : {[o.name for o in model.session.get_outputs()]}")
        ok()

        # ── 4. INFERENCE ──────────────────────────────────────────────────────
        section("4. inference")
        if model.is_seq2seq:
            _infer_s2s(model, spec)
        elif model_for == "ranker":
            _infer_ranker(model, spec)
        else:
            _infer_fe_sc(model, spec)

        # ── 5. CACHE HIT ──────────────────────────────────────────────────────
        section("5. cache hit")
        t0 = time.perf_counter()
        model2 = client.create_model(model_name, model_for=model_for)
        elapsed = time.perf_counter() - t0
        same = (model2.session is model.session) if not model.is_seq2seq else (model2.seq2seq_model is model.seq2seq_model)
        print(f"  elapsed      : {elapsed:.4f}s  (should be <0.01s)")
        print(f"  same session : {same}")
        ok() if same else warn("session was not reused")

        # ── 6. IS_LOADED ──────────────────────────────────────────────────────
        section("6. is_loaded()")
        loaded = client.is_loaded(model_name, model_for=model_for)
        print(f"  is_loaded    : {loaded}")
        ok() if loaded else print("  ✗  FAIL: is_loaded returned False")
        if not loaded:
            passed = False

        # ── 7. CACHE STATS ────────────────────────────────────────────────────
        section("7. cache_stats()")
        print(json.dumps(client.cache_stats(), indent=2))

        # ── 8. RELOAD ─────────────────────────────────────────────────────────
        section("8. reload()")
        t0 = time.perf_counter()
        reloaded = client.reload(model_name, model_for=model_for)
        elapsed = time.perf_counter() - t0
        print(f"  {reloaded}")
        print(f"  elapsed      : {elapsed:.3f}s")
        ok()

        # ── 9. UNLOAD ─────────────────────────────────────────────────────────
        section("9. unload()")
        evicted = client.unload(model_name, model_for=model_for)
        print(f"  evicted      : {evicted}")
        print(f"  is_loaded    : {client.is_loaded(model_name, model_for=model_for)}")
        ok() if evicted else warn("nothing evicted")

    except Exception as exc:
        print(f"\n  ✗  FAIL: {exc}")
        traceback.print_exc()
        passed = False

    return passed


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="floudsonnx unified manual smoke test — all model types",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Model types and their defaults:
  fe     : sentence-transformers/all-MiniLM-L6-v2  (feature-extraction)
  s2s    : google/flan-t5-small                     (seq2seq-lm)
  ranker : cross-encoder/ms-marco-MiniLM-L-12-v2   (sequence-classification)
  llm    : TinyLlama/TinyLlama-1.1B-Chat-v1.0      (text-generation-with-past)

Examples:
  python test_all_models.py
  python test_all_models.py --type fe s2s
  python test_all_models.py --type ranker --no-export
  python test_all_models.py --type llm --home C:/tmp/flouds
        """,
    )
    parser.add_argument(
        "--type", dest="types", nargs="+",
        choices=list(CATALOGUE.keys()),
        default=list(CATALOGUE.keys()),
        metavar="TYPE",
        help="Model type(s) to test: fe s2s ranker llm (default: all)",
    )
    parser.add_argument("--home", default="~/.flouds", help="Model store root")
    parser.add_argument("--no-export", action="store_true",
                        help="Skip pull; load existing models only")
    # Per-type model overrides
    parser.add_argument("--fe-model", default=None, metavar="MODEL",
                        help="Override fe model name")
    parser.add_argument("--s2s-model", default=None, metavar="MODEL",
                        help="Override s2s model name")
    parser.add_argument("--ranker-model", default=None, metavar="MODEL",
                        help="Override ranker model name")
    parser.add_argument("--llm-model", default=None, metavar="MODEL",
                        help="Override llm model name")
    args = parser.parse_args()

    # Apply model overrides
    overrides = {
        "fe": args.fe_model,
        "s2s": args.s2s_model,
        "ranker": args.ranker_model,
        "llm": args.llm_model,
    }
    for t, override in overrides.items():
        if override:
            CATALOGUE[t].model_name = override

    from floudsonnx import FloudsOnnxClient, FloudsOnnxSettings
    home = Path(args.home).expanduser()
    settings = FloudsOnnxSettings(home_dir=home, session_provider="CPUExecutionProvider")
    client = FloudsOnnxClient(settings)

    print(f"\nfloudsonnx unified model test")
    print(f"  store : {home}")
    print(f"  types : {args.types}")

    results: dict[str, bool] = {}
    for t in args.types:
        spec = CATALOGUE[t]
        results[t] = run_model_test(spec, client, no_export=args.no_export)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * 62}")
    print(f"  SUMMARY")
    print(f"{'=' * 62}")
    all_passed = True
    for t, passed in results.items():
        icon = "✓" if passed else "✗"
        status = "PASS" if passed else "FAIL"
        print(f"  {icon}  {t:8}  {CATALOGUE[t].model_name}  →  {status}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("  ALL TESTS PASSED")
        sys.exit(0)
    else:
        print("  SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
