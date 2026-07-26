# setup.ps1 — Windows setup for flm-tem-alignment
# Run from the repo root:  .\setup.ps1

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Write-Host "`n=== Step 1: Install uv ===" -ForegroundColor Cyan
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    irm https://astral.sh/uv/install.ps1 | iex
    # Reload PATH so uv is available in this session
    $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH", "User") + ";" +
                [System.Environment]::GetEnvironmentVariable("PATH", "Machine")
} else {
    Write-Host "  uv already installed"
}

Write-Host "`n=== Step 2: Install Python dependencies (uv sync) ===" -ForegroundColor Cyan
uv sync

Write-Host "`n=== Step 3: Clone model repos, download weights, install plugin ===" -ForegroundColor Cyan
uv run python install.py

Write-Host "`nSetup complete. Open napari to use the plugin." -ForegroundColor Green
