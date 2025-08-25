@echo off
echo 📦 Installing StockAI Dependencies...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8+ from python.org
    pause
    exit /b 1
)

echo ✅ Python found: 
python --version

echo.
echo 📦 Installing required packages...
pip install -r requirements.txt

echo.
echo 🔧 Installing additional required packages...
pip install jinja2 fastapi uvicorn python-multipart aiofiles

echo.
echo ✅ All dependencies installed successfully!
echo.
pause