# =============================================================================
# setup_dev.ps1 — One-shot dev environment setup for floudsonnx
# Run from the repo root: .\setup_dev.ps1
# =============================================================================

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot

Write-Host "`n=== floudsonnx dev setup ===" -ForegroundColor Cyan

# 1. Create venv if missing
if (-not (Test-Path "$Root\.venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv "$Root\.venv"
}

# 2. Activate
$activate = "$Root\.venv\Scripts\Activate.ps1"
if (Test-Path $activate) {
    & $activate
} else {
    Write-Error ".venv not found or activation script missing."
    exit 1
}

# 3. Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip --quiet

# 4. Install deps
Write-Host "Installing dependencies..." -ForegroundColor Yellow
pip install -r "$Root\requirements-dev.txt" --quiet
pip install -e "$Root[export,seq2seq]" --quiet

# 5. Install pre-commit hooks
Write-Host "Installing pre-commit hooks..." -ForegroundColor Yellow
pre-commit install
pre-commit install --hook-type pre-push

# 6. Verify
Write-Host "`nRunning pre-commit on all files..." -ForegroundColor Yellow
pre-commit run --all-files

Write-Host "`nRunning unit tests..." -ForegroundColor Yellow
pytest tests/unit/ -q

Write-Host "`n=== Setup complete ===" -ForegroundColor Green
Write-Host "Pre-commit hooks are installed and will run on every commit." -ForegroundColor Green
