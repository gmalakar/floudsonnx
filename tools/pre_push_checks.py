# =============================================================================
# File: pre_push_checks.py
# Date Created: 2026-06-10
# Date Updated: 2026-06-11
# Copyright (c) 2026 Goutam Malakar.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================
"""Run project sanity checks before push: pytest unit tests and mypy.
Exits non-zero if any step fails.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _venv_python(root: Path) -> Path:
    if os.name == "nt":
        return root / ".venv" / "Scripts" / "python.exe"
    return root / ".venv" / "bin" / "python"


def _reexec_with_venv_python() -> None:
    root = _repo_root()
    venv_python = _venv_python(root)
    current_python = Path(sys.executable).resolve()
    if venv_python.exists() and current_python != venv_python.resolve():
        os.execv(str(venv_python), [str(venv_python), *sys.argv])


_reexec_with_venv_python()

commands = [
    [sys.executable, "-m", "pytest", "tests/unit/", "-q"],
    [sys.executable, "-m", "pre_commit", "run", "-a", "mypy"],
]

for cmd in commands:
    print("Running:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Command failed (exit {e.returncode}): {' '.join(cmd)}")
        sys.exit(e.returncode)

print("All pre-push checks passed.")
sys.exit(0)
