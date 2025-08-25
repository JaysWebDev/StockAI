 @echo off
echo 🌙 Overnight StockAI Analysis Starting...
echo Current time: %date% %time%
echo.

REM Change to scheduler directory
cd /d D:\StockAI_Master\scheduler

REM Wait a bit for system to stabilize
timeout /t 5 /nobreak

REM Run the daily update
python daily_scheduler.py

echo.
echo 🌅 Overnight analysis completed at: %date% %time%
echo Check reports folder for results

REM Keep window open for 30 seconds to see results
timeout /t 30 /nobreak