@echo off
echo Creating daily scheduled task for StockAI...

REM Create a task that runs daily at 4:00 AM
schtasks /create /tn "StockAI Daily Update" /tr "D:\StockAI_Master\scheduler\overnight_runner.bat" /sc daily /st 04:00 /ru SYSTEM

echo.
echo ✅ Daily task scheduled to run at 4:00 AM every day
echo Task name: "StockAI Daily Update"
pause