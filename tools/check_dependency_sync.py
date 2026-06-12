# =============================================================================
# File: check_dependency_sync.py
# Date Created: 2026-06-10
# Date Updated: 2026-06-11
# Copyright (c) 2026 Goutam Malakar.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================
"""Check that requirements-prod.txt mirrors [project.dependencies] in pyproject.toml."""
from __future__ import annotations

import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read_requirements(path: Path) -> list[str]:
    reqs = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("-r "):
            continue
        reqs.append(line)
    return reqs


def main() -> int:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    pyproject_deps = list(pyproject["project"]["dependencies"])
    requirements_deps = _read_requirements(REPO_ROOT / "requirements-prod.txt")

    if pyproject_deps == requirements_deps:
        print("OK: requirements-prod.txt is in sync with pyproject.toml")
        return 0

    print("ERROR: requirements-prod.txt is out of sync with pyproject.toml", file=sys.stderr)
    print("\npyproject.toml [project.dependencies]:", file=sys.stderr)
    for d in pyproject_deps:
        print(f"  {d}", file=sys.stderr)
    print("\nrequirements-prod.txt:", file=sys.stderr)
    for d in requirements_deps:
        print(f"  {d}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
