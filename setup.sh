#!/usr/bin/env bash
# setup.sh — Mac/Linux setup for flm-tem-alignment
# Run from the repo root:  bash setup.sh

set -e

echo ""
echo "=== Step 1: Install uv ==="
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Make uv available in this session
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "  uv already installed"
fi

echo ""
echo "=== Step 2: Install Python dependencies (uv sync) ==="
uv sync

echo ""
echo "=== Step 3: Clone model repos, download weights, install plugin ==="
uv run python install.py

echo ""
echo "Setup complete. To launch napari run:"
echo "  uv run napari"
