@echo off
chcp 65001 >nul 2>&1
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ========================================
echo   Auto Grading System - Unit Tests
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
echo.

"!PYTHON_CMD!" -X utf8 -m pytest tests/ -v --tb=short > test_report.txt 2>&1
type test_report.txt

echo.
pause
