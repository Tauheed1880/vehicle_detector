@echo off
rem ==============================================================================
rem Automated Installer for Windows
rem Shows live download progress bar
rem Non-interactive and Idempotent
rem ==============================================================================

echo === FastALPR + Vehicle Tracking Installer (Windows) ===

rem 1. Check Python
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python not found in PATH. Please install Python 3.10+ and add it to PATH.
    exit /b 1
)

rem 2. Upgrade pip
echo [1/4] Checking pip...
python -m pip install --upgrade pip --no-input

rem 3. Detect GPU
where nvidia-smi >nul 2>nul
if %errorlevel% equ 0 (
    echo [2/4] NVIDIA GPU detected. Installing onnxruntime-gpu...
    python -m pip uninstall -y onnxruntime >nul 2>&1
    python -m pip install onnxruntime-gpu>=1.19.2 --no-input
) else (
    echo [2/4] No NVIDIA GPU detected. Installing CPU onnxruntime...
    python -m pip uninstall -y onnxruntime-gpu >nul 2>&1
    python -m pip install onnxruntime>=1.19.2 --no-input
)

rem 4. Install dependencies
echo [3/4] Installing requirements.txt (ultralytics, fast-plate-ocr, etc.)...
python -m pip install -r requirements.txt --no-input

rem 5. Install local package
echo [4/4] Installing local fast-alpr package...
python -m pip install -e . --no-deps --no-input

echo =================================================
echo === Installation Completed Successfully! ===
echo =================================================
