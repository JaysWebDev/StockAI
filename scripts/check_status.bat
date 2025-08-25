REM check_status.bat
@echo off
echo 🔍 StockAI_Master System Status Check
echo =====================================
echo.
if exist "data\unified_stock_intelligence.db" (
    echo ✅ Database: Found
) else (
    echo ❌ Database: Missing
)

if exist "config.yaml" (
    echo ✅ Config: Found
) else (
    echo ❌ Config: Missing  
)

python -c "import pandas, yfinance, requests, yaml; print('✅ Dependencies: All installed')" 2>nul || echo ❌ Dependencies: Missing packages

echo.
echo 📊 Recent Reports:
dir data\daily_report_*.txt /B 2>nul | head -3

pause