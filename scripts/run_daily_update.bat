@echo off
REM run_daily_update.bat - Run daily analysis (Updated paths)
echo 📊 Running Daily Stock Analysis...
echo.

REM Navigate to project root
cd /d "%~dp0\.."

REM Check if system files exist
if not exist "backend\unified_stock_intelligence.py" (
    echo ❌ Main system file not found!
    echo Please ensure unified_stock_intelligence.py is in backend directory
    pause
    exit /b 1
)

if not exist "scheduler\daily_scheduler.py" (
    echo ❌ Scheduler file not found!
    echo Please ensure daily_scheduler.py is in scheduler directory
    pause
    exit /b 1
)

echo 🔄 Starting comprehensive daily update...
echo This may take 5-10 minutes depending on your watchlist size
echo.

python scheduler\daily_scheduler.py

echo.
echo ✅ Daily update completed!
echo Check the data folder for reports and logs
echo.
pause