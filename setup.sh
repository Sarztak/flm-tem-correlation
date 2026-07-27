#!/usr/bin/env bash
# setup.sh — Mac/Linux setup for flm-tem-alignment
# Run from the repo root:  bash setup.sh

set -e

echo ""
echo "=== Step 1: Install uv ==="
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "  uv already installed"
fi

echo ""
echo "=== Step 2: Clone model repos ==="
uv run --no-sync python install.py clone

echo ""
echo "=== Step 3: Install Python dependencies ==="
uv pip install setuptools
uv sync

echo ""
echo "=== Step 4: Download model weights ==="
uv run python install.py weights

echo ""
echo "Setup complete. To launch napari run:"
echo "  uv run napari"
