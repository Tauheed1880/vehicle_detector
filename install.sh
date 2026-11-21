#!/usr/bin/env bash
# ==============================================================================
# Automated Fast Installer for Linux / Google Colab
# Uses --prefer-binary to prevent slow source compilation / backend dependency builds
# ==============================================================================

set -e

echo "=== FastALPR + Vehicle Tracking Fast Installer ==="

# 1. Check Python version
PYTHON_CMD="python3"
if ! command -v python3 &> /dev/null; then
    PYTHON_CMD="python"
fi
echo "[INFO] Using Python: $($PYTHON_CMD --version)"

# 2. Upgrade pip
echo "[1/3] Checking pip..."
$PYTHON_CMD -m pip install --upgrade pip --no-input

# 3. Detect NVIDIA GPU / CUDA
HAS_GPU=false
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        HAS_GPU=true
    fi
fi

if [ "$HAS_GPU" = true ]; then
    echo "[2/3] NVIDIA GPU detected. Installing pre-built onnxruntime-gpu..."
    $PYTHON_CMD -m pip uninstall -y onnxruntime 2>/dev/null || true
    $PYTHON_CMD -m pip install onnxruntime-gpu>=1.19.2 --no-input --prefer-binary
else
    echo "[2/3] No NVIDIA GPU detected. Installing pre-built CPU onnxruntime..."
    $PYTHON_CMD -m pip uninstall -y onnxruntime-gpu 2>/dev/null || true
    $PYTHON_CMD -m pip install onnxruntime>=1.19.2 --no-input --prefer-binary
fi

# 4. Install dependencies (instant pre-built binary wheels)
echo "[3/3] Installing dependencies with pre-built binary wheels..."
$PYTHON_CMD -m pip install -r requirements.txt --no-input --prefer-binary

echo "================================================="
echo "=== Installation Completed Successfully! ==="
echo "================================================="
