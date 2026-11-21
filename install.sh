#!/usr/bin/env bash
# ==============================================================================
# Automated Installer for Linux / Google Colab
# Shows live download percentage and progress bar
# Non-interactive (-y flags) and Idempotent
# ==============================================================================

set -e

echo "=== FastALPR + Vehicle Tracking Installer (Linux / Colab) ==="

# 1. Check Python version
PYTHON_CMD="python3"
if ! command -v python3 &> /dev/null; then
    PYTHON_CMD="python"
fi
echo "[INFO] Using Python: $($PYTHON_CMD --version)"

# 2. Upgrade pip
echo "[1/4] Checking pip..."
$PYTHON_CMD -m pip install --upgrade pip --no-input

# 3. Detect NVIDIA GPU / CUDA
HAS_GPU=false
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        HAS_GPU=true
    fi
fi

if [ "$HAS_GPU" = true ]; then
    echo "[2/4] NVIDIA GPU detected. Installing onnxruntime-gpu..."
    # Ensure CPU onnxruntime is removed first to avoid provider conflicts
    $PYTHON_CMD -m pip uninstall -y onnxruntime 2>/dev/null || true
    $PYTHON_CMD -m pip install onnxruntime-gpu>=1.19.2 --no-input
else
    echo "[2/4] No NVIDIA GPU detected. Installing CPU onnxruntime..."
    $PYTHON_CMD -m pip uninstall -y onnxruntime-gpu 2>/dev/null || true
    $PYTHON_CMD -m pip install onnxruntime>=1.19.2 --no-input
fi

# 4. Install requirements.txt with visible progress bar
echo "[3/4] Installing dependencies from requirements.txt (ultralytics, fast-plate-ocr, etc.)..."
$PYTHON_CMD -m pip install -r requirements.txt --no-input

# 5. Install fast-alpr in editable mode
echo "[4/4] Installing local fast-alpr package..."
$PYTHON_CMD -m pip install -e . --no-deps --no-input

echo "================================================="
echo "=== Installation Completed Successfully! ==="
echo "================================================="
