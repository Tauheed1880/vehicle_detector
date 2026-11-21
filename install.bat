@echo off
rem ==============================================================================
rem Automated Installer for Windows
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
python -m pip install --upgrade pip --no-input -q

rem 3. Detect GPU
where nvidia-smi >nul 2>nul
if %errorlevel% equ 0 (
    echo [INFO] NVIDIA GPU detected. Installing onnxruntime-gpu...
    python -m pip uninstall -y onnxruntime >nul 2>&1
    python -m pip install onnxruntime-gpu>=1.19.2 --no-input -q
) else (
    echo [INFO] No NVIDIA GPU detected. Installing CPU onnxruntime...
    python -m pip uninstall -y onnxruntime-gpu >nul 2>&1
    python -m pip install onnxruntime>=1.19.2 --no-input -q
)

rem 4. Install dependencies
echo [INFO] Installing requirements.txt...
python -m pip install -r requirements.txt --no-input -q

rem 5. Install local package
echo [INFO] Installing local fast-alpr package...
python -m pip install -e . --no-deps --no-input -q

echo === Installation Completed Successfully! ===
