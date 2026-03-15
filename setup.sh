#!/bin/bash
set -e
# =============================================================================
# LTX-2 LoRA Training Bootstrap for RunPod
#
# Usage: Start a stock PyTorch pod, then run:
#   curl -sL https://raw.githubusercontent.com/Armychimp/ltx2-train-pod-build/main/setup.sh | bash
#
# Or set as Container Start Command for auto-start:
#   bash -c "curl -sL https://raw.githubusercontent.com/Armychimp/ltx2-train-pod-build/main/setup.sh | bash"
#
# Required env vars (set in RunPod pod template):
#   S3_ENDPOINT_URL, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, HF_TOKEN
#
# Training env vars (set for auto-start, or omit for manual mode):
#   DATASET, CONFIG, PREPROCESS, RESUME, STEPS, TERMINATE, RESOLUTION
#
# No secrets in this script — everything comes from env vars.
# =============================================================================

WORKSPACE="/workspace"
LTX2_DIR="$WORKSPACE/LTX-2"
SCRIPT_URL="https://raw.githubusercontent.com/Armychimp/ltx2-train-pod-build/main/train.py"

echo "============================================"
echo "LTX-2 Training Bootstrap"
echo "============================================"

# ---- Step 1: Install LTX-2 (cached in /workspace) ----
if [ ! -f "$LTX2_DIR/.venv/bin/python" ]; then
    echo ""
    echo "=== Installing LTX-2 trainer (first run, cached for future) ==="

    # System deps
    if ! command -v ffmpeg &>/dev/null; then
        echo "  Installing system packages..."
        apt-get update -qq && apt-get install -y -qq ffmpeg libgl1-mesa-glx libglib2.0-0 > /dev/null 2>&1
    fi

    # uv
    if ! command -v uv &>/dev/null; then
        echo "  Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1
        export PATH="$HOME/.local/bin:$PATH"
    fi

    # Clone and install
    if [ ! -d "$LTX2_DIR" ]; then
        echo "  Cloning LTX-2..."
        git clone --depth 1 https://github.com/Lightricks/LTX-2.git "$LTX2_DIR"
    fi

    echo "  Installing Python dependencies (~2-3 min)..."
    cd "$LTX2_DIR" && uv sync --frozen
    cd "$WORKSPACE"

    echo "  Done."
else
    echo "LTX-2 trainer already installed."
    export PATH="$HOME/.local/bin:$PATH"
fi

# ---- Step 2: Install pip deps for train.py ----
echo ""
echo "=== Checking pip dependencies ==="
pip install -q boto3 "huggingface_hub[hf_transfer]" pyyaml 2>/dev/null

# ---- Step 3: Download latest train.py ----
echo ""
echo "=== Downloading train.py ==="
curl -sL "$SCRIPT_URL" -o "$WORKSPACE/train.py"
echo "  Done."

# ---- Step 4: Run or wait ----
if [ -n "$DATASET" ] && [ -n "$CONFIG" ]; then
    echo ""
    echo "=== Starting training ==="
    python3 -u "$WORKSPACE/train.py"
    # If we get here, train.py exited (success, failure, or killed).
    # Sleep forever to prevent the container from restarting and looping.
    # The pod will be stopped by train.py's self-terminate, or manually.
    echo ""
    echo "Training script exited. Sleeping to prevent container restart loop."
    echo "Stop the pod manually if it didn't self-terminate."
    sleep infinity
else
    echo ""
    echo "============================================"
    echo "No DATASET/CONFIG set. Manual mode."
    echo ""
    echo "To start training:"
    echo "  DATASET=bp_rework CONFIG=bp_rework_lora python3 -u /workspace/train.py"
    echo ""
    echo "Or with all options:"
    echo "  DATASET=bp_rework CONFIG=bp_rework_lora PREPROCESS=true RESUME=false STEPS=4000 python3 -u /workspace/train.py"
    echo "============================================"
fi
