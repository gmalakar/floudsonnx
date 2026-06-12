# =============================================================================
# File: add_all_headers.py
# Date Created: 2026-06-10
# Date Updated: 2026-06-11
# Copyright (c) 2026 Goutam Malakar.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================
"""Add copyright/SPDX header to all Python source files that are missing one."""

import glob
import sys
from datetime import datetime
from pathlib import Path

HEADER = """# =============================================================================
# File: {filename}
# Date Created: {date_created}
# Date Updated: {date_updated}
# Copyright (c) {year} Goutam Malakar.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================
"""

EXCLUDES = {
    ".venv",
    "__pycache__",
    ".git",
    "build",
    "dist",
    ".egg-info",
    ".vs",
    ".vscode",
    ".pytest_cache",
}


def has_header(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8")
        return "Copyright (c)" in text or "SPDX-License-Identifier" in text
    except Exception:
        return False


def add_header_to_file(path: Path) -> None:
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    header = HEADER.format(
        filename=path.name,
        date_created=date_str,
        date_updated=date_str,
        year=now.year,
    )
    content = path.read_text(encoding="utf-8")
    path.write_text(header + "\n" + content, encoding="utf-8")


def iter_python_files(paths: list[str]) -> list[Path]:
    if paths:
        return [Path(p) for p in paths if Path(p).suffix == ".py"]
    return [Path(p) for p in glob.glob("**/*.py", recursive=True)]


def main(paths: list[str]) -> int:
    errors = 0
    for p in iter_python_files(paths):
        if any(part in EXCLUDES for part in p.parts):
            continue
        if has_header(p):
            continue
        try:
            add_header_to_file(p)
        except Exception as e:
            errors += 1
            print(f"Error adding header to {p}: {e}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
