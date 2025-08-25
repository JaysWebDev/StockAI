REM backup_data.bat
@echo off
set BACKUP_DIR=D:\StockAI_Master_Backup_%date:~-4,4%%date:~-10,2%%date:~-7,2%
mkdir %BACKUP_DIR%
xcopy data %BACKUP_DIR%\data /E /I /Y
xcopy logs %BACKUP_DIR%\logs /E /I /Y
echo ✅ Backup completed at %BACKUP_DIR%
pause