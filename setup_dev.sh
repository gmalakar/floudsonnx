#!/usr/bin/env bash
# =============================================================================
# setup_dev.sh — One-shot dev environment setup for floudsonnx (Linux/macOS)
# Run from the repo root: ./setup_dev.sh
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ""
echo "=== floudsonnx dev setup ==="

# 1. Create venv if missing
if [ ! -d "$ROOT/.venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$ROOT/.venv"
fi

# 2. Activate
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"

# 3. Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip --quiet

# 4. Install deps
echo "Installing dependencies..."
pip install -r "$ROOT/requirements-dev.txt" --quiet
pip install -e "$ROOT[export,seq2seq]" --quiet

# 5. Install pre-commit hooks
echo "Installing pre-commit hooks..."
pre-commit install
pre-commit install --hook-type pre-push

# 6. Verify
echo ""
echo "Running pre-commit on all files..."
pre-commit run --all-files

echo ""
echo "Running unit tests..."
pytest tests/unit/ -q

echo ""
echo "=== Setup complete ==="
echo "Pre-commit hooks are installed and will run on every commit."
