#!/usr/bin/env bash
# ==============================================================================
# Automated Installer for Linux / Google Colab
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

# 2. Upgrade pip silently
$PYTHON_CMD -m pip install --upgrade pip --no-input -q

# 3. Detect NVIDIA GPU / CUDA
HAS_GPU=false
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        HAS_GPU=true
    fi
fi

if [ "$HAS_GPU" = true ]; then
    echo "[INFO] NVIDIA GPU detected. Installing GPU-accelerated ONNX Runtime..."
    # Ensure CPU onnxruntime is removed first to avoid provider conflicts
    $PYTHON_CMD -m pip uninstall -y onnxruntime -q 2>/dev/null || true
    $PYTHON_CMD -m pip install onnxruntime-gpu>=1.19.2 --no-input -q
else
    echo "[INFO] No NVIDIA GPU detected. Installing CPU ONNX Runtime..."
    $PYTHON_CMD -m pip uninstall -y onnxruntime-gpu -q 2>/dev/null || true
    $PYTHON_CMD -m pip install onnxruntime>=1.19.2 --no-input -q
fi

# 4. Install requirements.txt
echo "[INFO] Installing dependencies from requirements.txt..."
$PYTHON_CMD -m pip install -r requirements.txt --no-input -q

# 5. Install fast-alpr in editable mode
echo "[INFO] Installing local fast-alpr package..."
$PYTHON_CMD -m pip install -e . --no-deps --no-input -q

echo "=== Installation Completed Successfully! ==="
