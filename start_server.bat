@echo off
echo 🚀 Starting StockAI Server...
cd /d D:\StockAI_Master\backend
start "StockAI Server" python main.py
echo ✅ Server started! Open: http://localhost:8000
pause