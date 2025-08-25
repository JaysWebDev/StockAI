
REM FILE 1: setup.bat
@echo off
REM setup.bat - Initial setup for StockAI_Master
echo 🚀 Setting up StockAI_Master System...
echo.

REM Create directory structure
echo 📁 Creating directory structure...
if not exist "data" mkdir data
if not exist "logs" mkdir logs
if not exist "archive" mkdir archive

REM Check Python installation
echo 🐍 Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found! Please install Python 3.8+ first.
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Install required packages
echo 📦 Installing required packages...
pip install pandas>=2.0.0
pip install yfinance>=0.2.0
pip install requests>=2.28.0
pip install pyyaml>=6.0

REM Test basic functionality
echo 🧪 Testing basic functionality...
python -c "import pandas; import yfinance; import requests; import yaml; print('✅ All packages imported successfully')"

if errorlevel 1 (
    echo ❌ Package installation failed!
    echo Please check your Python/pip installation
    pause
    exit /b 1
)

echo.
echo ✅ Setup completed successfully!
echo.
echo 📋 Next steps:
echo   1. Edit config.yaml to customize settings
echo   2. Run 'run_system.bat' to start the system
echo   3. Run 'run_daily_update.bat' to process your watchlist
echo.
pause