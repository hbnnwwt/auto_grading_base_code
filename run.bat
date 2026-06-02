@echo off
chcp 65001 >nul 2>&1
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ========================================
echo   Auto Grading System
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
echo   [1] Single image
echo   [2] Batch grading (data\answer_sheets)
echo.
set /p "MODE=Select mode [1/2]: "

if "!MODE!"=="2" (
    echo.
    echo [Mode] Batch grading
    echo.
    "!PYTHON_CMD!" -X utf8 main.py --folder "data\answer_sheets"
) else (
    if "%~1"=="" (
        set "IMAGE=data\answer_sheets\answer_sheet_1.png"
        echo [Info] No image specified, using default: !IMAGE!
        echo   You can also drag an image file onto this script.
    ) else (
        set "IMAGE=%~1"
    )
    if not exist "!IMAGE!" (
        echo [Error] Image not found: !IMAGE!
        echo.
        pause
        exit /b 1
    )
    echo [Mode] Single image
    echo [Image] !IMAGE!
    echo.
    "!PYTHON_CMD!" -X utf8 main.py --image "!IMAGE!"
)

echo.
pause
