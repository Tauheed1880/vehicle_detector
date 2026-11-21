@echo off
rem ==============================================================================
rem Automated Fast Installer for Windows
rem Uses --prefer-binary to avoid slow compilation
rem ==============================================================================

echo === FastALPR + Vehicle Tracking Fast Installer ===

rem 1. Check Python
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python not found in PATH. Please install Python 3.10+ and add it to PATH.
    exit /b 1
)

rem 2. Upgrade pip
echo [1/3] Checking pip...
python -m pip install --upgrade pip --no-input

rem 3. Detect GPU
where nvidia-smi >nul 2>nul
if %errorlevel% equ 0 (
    echo [2/3] NVIDIA GPU detected. Installing onnxruntime-gpu...
    python -m pip uninstall -y onnxruntime >nul 2>&1
    python -m pip install onnxruntime-gpu>=1.19.2 --no-input --prefer-binary
) else (
    echo [2/3] No NVIDIA GPU detected. Installing CPU onnxruntime...
    python -m pip uninstall -y onnxruntime-gpu >nul 2>&1
    python -m pip install onnxruntime>=1.19.2 --no-input --prefer-binary
)

rem 4. Install dependencies
echo [3/3] Installing requirements.txt with pre-built binary wheels...
python -m pip install -r requirements.txt --no-input --prefer-binary

echo =================================================
echo === Installation Completed Successfully! ===
echo =================================================
