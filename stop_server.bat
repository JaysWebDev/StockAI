@echo off
echo ⏹️ Stopping StockAI Server...
taskkill /f /im python.exe /t 2>nul
taskkill /f /im uvicorn.exe /t 2>nul
echo ✅ All StockAI processes stopped
pause