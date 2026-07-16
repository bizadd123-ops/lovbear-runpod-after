#!/usr/bin/env bash
set -euo pipefail
cd /workspace
echo "[lovbear] installing deps..."
pip install -q --no-cache-dir "runpod>=1.7.0" "diffusers>=0.30.0" transformers accelerate safetensors Pillow
HANDLER_URL="${HANDLER_URL:-https://raw.githubusercontent.com/bizadd123-ops/lovbear-runpod-after/main/handler.py}"
echo "[lovbear] fetching handler from $HANDLER_URL"
curl -fsSL "$HANDLER_URL" -o /workspace/handler.py
echo "[lovbear] starting serverless handler"
python -u /workspace/handler.py
