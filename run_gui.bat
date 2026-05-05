@echo off
chcp 65001 >nul 2>&1
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ========================================
echo   Auto Grading System - GUI
echo ========================================
echo.

set "PYTHON_CMD="

if exist "python_portable\python.exe" (
    set "PYTHON_CMD=python_portable\python.exe"
) else (
    py --version >nul 2>&1
    if !errorlevel! equ 0 (
        set "PYTHON_CMD=py"
    ) else (
        python --version >nul 2>&1
        if !errorlevel! equ 0 (
            set "PYTHON_CMD=python"
        ) else (
            echo [Error] Python not found.
            echo   Please run setup.bat first, or install Python 3.8+.
            echo.
            pause
            exit /b 1
        )
    )
)

echo [Using] !PYTHON_CMD!
echo [Launching] Streamlit GUI...
echo.

"!PYTHON_CMD!" -X utf8 -m streamlit run app.py --server.port 8501

if !errorlevel! neq 0 (
    echo.
    echo [Error] Failed to start Streamlit.
    echo   Try: !PYTHON_CMD! -m pip install streamlit
    echo.
)

pause
