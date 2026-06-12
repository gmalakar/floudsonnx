# =============================================================================
# Copyright (c) 2026 Goutam Malakar. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# =============================================================================
"""
floudsonnx CLI — entry-point: floudsonnx <command> [options]

Commands:
  pull    <model_name>  [--for fe] [--task T] [--optimize] [--force]
                        [--trust-remote-code] [--use-external-data-format]
                        [--use-fallback-if-failed] [--hf-token T]
  list
  info    <model_name>  [--for fe]
  remove  <model_name>  [--for fe]
  reload  <model_name>  [--for fe]
  stats
  serve   [--host H] [--port P]
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any


def _client() -> Any:
    from floudsonnx.api.client import get_default_client

    return get_default_client()


def cmd_pull(args: argparse.Namespace) -> int:
    print(f"Pulling '{args.model_name}' (model_for={args.model_for}, optimize={args.optimize}) ...")
    try:
        manifest = _client().pull(
            args.model_name,
            model_for=args.model_for,
            task=args.task or None,
            force=args.force,
            optimize=args.optimize,
            trust_remote_code=args.trust_remote_code,
            use_external_data_format=args.use_external_data_format,
            use_fallback_if_failed=args.use_fallback_if_failed,
            hf_token=args.hf_token or None,
        )
        print(f"OK  {manifest.model_name}  [{manifest.model_for}]  pulled_at={manifest.pulled_at}")
        print(f"    strategy : {manifest.session_strategy}")
        print(f"    onnx     : {', '.join(manifest.onnx_files) or 'none'}")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


def cmd_list(args: argparse.Namespace) -> int:
    manifests = _client().list()
    if not manifests:
        print("No models in local store.")
        return 0
    fmt = "{:<50}  {:<10}  {:<25}  {}"
    print(fmt.format("MODEL", "FOR", "PULLED AT", "STRATEGY"))
    print("-" * 105)
    for m in manifests:
        print(fmt.format(m.model_name[:48], m.model_for, m.pulled_at[:24], m.session_strategy))
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    manifest = _client()._loader._registry.get_manifest(args.model_name, args.model_for)
    if manifest is None:
        print(f"Model '{args.model_name}' not found.", file=sys.stderr)
        return 1
    print(json.dumps(manifest.model_dump(mode="json"), indent=2))
    return 0


def cmd_remove(args: argparse.Namespace) -> int:
    removed = _client().remove(args.model_name, args.model_for)
    print(f"Removed '{args.model_name}'" if removed else f"'{args.model_name}' was not in the local store.")
    return 0


def cmd_reload(args: argparse.Namespace) -> int:
    try:
        loaded = _client().reload(args.model_name, args.model_for)
        print(f"Reloaded '{loaded.model_name}' — strategy={loaded.session_strategy.value}")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


def cmd_stats(args: argparse.Namespace) -> int:
    print(json.dumps(_client().cache_stats(), indent=2))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        from floudsonnx.api.server import run_server

        print(f"Starting floudsonnx server on {args.host}:{args.port} ...")
        run_server(host=args.host, port=args.port)
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="floudsonnx", description="Ollama-style ONNX model store and runtime")
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    sub.required = True

    # pull
    p = sub.add_parser("pull", help="Pull (export) a model to the local store")
    p.add_argument("model_name")
    p.add_argument("--for", dest="model_for", default="fe", help="fe|s2s|sc|ranker|llm (default: fe)")
    p.add_argument("--task", default="", help="Optimum export task")
    p.add_argument("--optimize", action="store_true", default=True)
    p.add_argument("--no-optimize", dest="optimize", action="store_false")
    p.add_argument("--force", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--use-external-data-format", action="store_true")
    p.add_argument("--use-fallback-if-failed", action="store_true")
    p.add_argument("--hf-token", default="", help="HuggingFace API token")
    p.set_defaults(func=cmd_pull)

    # list
    p = sub.add_parser("list", help="List all locally stored models")
    p.set_defaults(func=cmd_list)

    # info
    p = sub.add_parser("info", help="Show manifest for a model")
    p.add_argument("model_name")
    p.add_argument("--for", dest="model_for", default="fe")
    p.set_defaults(func=cmd_info)

    # remove
    p = sub.add_parser("remove", help="Delete a model from the local store")
    p.add_argument("model_name")
    p.add_argument("--for", dest="model_for", default="fe")
    p.set_defaults(func=cmd_remove)

    # reload
    p = sub.add_parser("reload", help="Evict and re-load a model session from disk")
    p.add_argument("model_name")
    p.add_argument("--for", dest="model_for", default="fe")
    p.set_defaults(func=cmd_reload)

    # stats
    p = sub.add_parser("stats", help="Show session cache statistics")
    p.set_defaults(func=cmd_stats)

    # serve
    p = sub.add_parser("serve", help="Start the HTTP server")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=19720)
    p.set_defaults(func=cmd_serve)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
