@echo off
setlocal

echo.
echo === Step 1: Install uv ===
where uv >nul 2>&1
if %errorlevel% neq 0 (
    powershell -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    set "PATH=%USERPROFILE%\.local\bin;%PATH%"
) else (
    echo   uv already installed
)

echo.
echo === Step 2: Install Python dependencies (uv sync) ===
uv sync
if %errorlevel% neq 0 (
    echo [ERROR] uv sync failed
    exit /b 1
)

echo.
echo === Step 3: Clone model repos, download weights, install plugin ===
uv run python install.py
if %errorlevel% neq 0 (
    echo [ERROR] install.py failed
    exit /b 1
)

echo.
echo Setup complete. To launch napari run:
echo   uv run napari
