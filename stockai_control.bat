@echo off
chcp 65001 >nul
title StockAI Master Control
echo.
echo ========================================
echo          STOCKAI MASTER CONTROL
echo ========================================
echo.

:menu
cls
echo.
echo 🎯 STOCKAI MASTER CONTROL PANEL
echo ================================
echo.
echo 1. 🚀 START StockAI (Auto + Daily Updates)
echo 2. ⚡ START Server Only
echo 3. 🔄 Manual Data Refresh
echo 4. ⏹️  STOP StockAI System
echo 5. 📊 Check System Status
echo 6. 🛠️  Check/Install Dependencies
echo 7. ❌ Exit
echo.
set /p choice="Choose an option (1-7): "

if "%choice%"=="1" goto start_auto
if "%choice%"=="2" goto start_server
if "%choice%"=="3" goto manual_refresh
if "%choice%"=="4" goto stop
if "%choice%"=="5" goto status
if "%choice%"=="6" goto check_deps
if "%choice%"=="7" goto exit

echo Invalid choice. Please try again.
timeout /t 2 /nobreak >nul
goto menu

:start_auto
echo.
echo 🚀 Starting StockAI with Auto-Updates...
echo 📊 Dashboard: http://localhost:8000
echo ⏰ Daily updates at 4:00 AM
echo.
cd /d D:\StockAI_Master\backend
start "StockAI Auto" python auto_start.py
timeout /t 3 /nobreak >nul
echo ✅ StockAI started with auto-updates!
echo.
pause
goto menu

:start_server
echo.
echo ⚡ Starting StockAI Server Only...
echo 📊 Dashboard: http://localhost:8000
echo.
cd /d D:\StockAI_Master\backend
start "StockAI Server" python main.py
timeout /t 3 /nobreak >nobreak
echo ✅ StockAI server started!
echo.
pause
goto menu

:manual_refresh
echo.
echo 🔄 Manual Data Refresh...
echo ⏳ This may take a few minutes...
echo.
cd /d D:\StockAI_Master\scheduler
python daily_scheduler.py
echo.
echo ✅ Manual refresh completed!
echo.
pause
goto menu

:stop
echo.
echo ⏹️ Stopping StockAI System...
taskkill /f /im python.exe /t 2>nul
taskkill /f /im uvicorn.exe /t 2>nul
echo ✅ All StockAI processes stopped
timeout /t 2 /nobreak >nul
goto menu

:status
echo.
echo 📊 Checking StockAI Status...
echo.
tasklist /fi "imagename eq python.exe" | find /i "python" >nul
if %errorlevel%==0 (
    echo ✅ StockAI is RUNNING
    echo 🌐 Dashboard: http://localhost:8000
    echo 🔗 Health: http://localhost:8000/health
) else (
    echo ❌ StockAI is STOPPED
)

timeout /t 3 /nobreak >nul
echo.
pause
goto menu

:check_deps
echo.
echo 🛠️ Checking Dependencies...
echo.
cd /d D:\StockAI_Master\backend
python check_dependencies.py
echo.
pause
goto menu

:exit
echo.
echo 👋 Thank you for using StockAI!
echo.
timeout /t 2 /nobreak >nul
exit