@echo off
REM run_system.bat - Start the main system (Updated paths)
echo 🚀 Starting StockAI_Master System...
echo.

REM Navigate to project root
cd /d "%~dp0\.."

REM Check if config exists
if not exist "config\config.yaml" (
    echo ⚠️ config.yaml not found in config directory!
    pause
    exit /b 1
)

echo 🌐 Starting web system...
echo Dashboard will open at: http://localhost:8000
echo Press Ctrl+C to stop the system
echo.

python backend\unified_stock_intelligence.py

echo.
echo 👋 System stopped
pause