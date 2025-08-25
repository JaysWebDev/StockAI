@echo off
echo Enabling StockAI Auto-Start on Boot...
echo.

REM Create a startup task in Windows Task Scheduler
schtasks /create /tn "StockAI Auto Start" /tr "D:\StockAI_Master\stockai_control.bat start_auto" /sc onlogon /ru %USERNAME% /rl highest

echo.
echo ✅ StockAI will now start automatically when you log in!
echo 📊 Dashboard: http://localhost:8000
echo ⏰ Daily updates: 4:00 AM
echo.
pause