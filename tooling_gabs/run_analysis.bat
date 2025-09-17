@echo off
echo ============================================================
echo LOCAL AUTOFORMER EBM ANALYSIS
echo ============================================================
echo.

echo Checking Python installation...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

echo.
echo Running analysis...
python run_local_analysis.py

echo.
echo Analysis completed. Press any key to exit.
pause

