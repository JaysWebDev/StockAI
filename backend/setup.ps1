# StockAI_Master Setup Script
# Run this script from D:\StockAI_Master\backend or adjust the paths accordingly

# Set execution policy (may require admin rights)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force

# Display project structure
Write-Host "StockAI_Master Project Structure" -ForegroundColor Green
Write-Host "=================================" -ForegroundColor Green
Get-ChildItem -Path "D:\StockAI_Master" | Format-Table Name, LastWriteTime, @{Name="Type";Expression={if($_.PSIsContainer){"Folder"}else{"File"}}}, Length

# Check if Python is installed
Write-Host "`nChecking Python installation..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "Python found: $pythonVersion" -ForegroundColor Green
} else {
    Write-Host "Python not found. Please install Python 3.8+ from python.org" -ForegroundColor Red
    exit 1
}

# Check if pip is installed
Write-Host "Checking pip installation..." -ForegroundColor Yellow
$pipVersion = pip --version 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "Pip found: $($pipVersion[0])" -ForegroundColor Green
} else {
    Write-Host "Pip not found. Installing pip..." -ForegroundColor Yellow
    python -m ensurepip --upgrade
}

# Navigate to backend directory
Set-Location "D:\StockAI_Master\backend"

# Install required packages
Write-Host "`nInstalling required Python packages..." -ForegroundColor Yellow
pip install -r ../requirements.txt

# Create necessary directories if they don't exist
Write-Host "`nCreating directory structure..." -ForegroundColor Yellow
$directories = @("logs", "data", "config", "reports")
foreach ($dir in $directories) {
    if (!(Test-Path "../$dir")) {
        New-Item -ItemType Directory -Path "../$dir" -Force | Out-Null
        Write-Host "Created directory: ../$dir" -ForegroundColor Green
    }
}

# Create default config file if it doesn't exist
$configPath = "../config/config.yaml"
if (!(Test-Path $configPath)) {
    Write-Host "Creating default config file..." -ForegroundColor Yellow
    $configContent = @"
# StockAI Configuration
alpha_vantage_api_key: "DQ4I233CG66GOW9T"
email: "jaysweb@proton.me"
db_path: "data/unified_stock_intelligence.db"
log_level: "INFO"
web_port: 8000
rate_limit_delay: 12
max_retries: 3

# Watchlist settings
default_watchlist:
  - "AAPL"
  - "MSFT"
  - "GOOGL"
  - "AMZN"
  - "TSLA"

# Email alert settings
email_alerts: false
smtp_server: ""
smtp_port: 587
email_user: ""
email_password: ""
alert_recipients: []

# Analysis settings
update_frequency: "daily"
batch_size: 10
technical_indicators:
  - "sma_50"
  - "sma_200"
  - "rsi_14"
  - "macd"
  - "bollinger_bands"
"@
    Set-Content -Path $configPath -Value $configContent
    Write-Host "Created config file: $configPath" -ForegroundColor Green
}

# Check if the main database file exists
$dbPath = "../data/unified_stock_intelligence.db"
if (Test-Path $dbPath) {
    Write-Host "Database file found: $dbPath" -ForegroundColor Green
    $dbSize = (Get-Item $dbPath).Length / 1KB
    Write-Host "Database size: $([math]::Round($dbSize, 2)) KB" -ForegroundColor Green
} else {
    Write-Host "Database file not found. It will be created when the application runs." -ForegroundColor Yellow
}

# Create a startup script for the application
$startScript = @"
#!/usr/bin/env python3
"""
Unified Stock Intelligence Platform Startup Script
"""
import os
import sys
import logging
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Change to backend directory
os.chdir(Path(__file__).parent)

try:
    from backend.unified_stock_platform import UnifiedStockIntelligence, Config
    
    print("🚀 Starting Unified Stock Intelligence Platform...")
    print("📍 Running from:", os.getcwd())
    
    # Load configuration
    config = Config.load_from_file("../config/config.yaml")
    
    # Initialize and run the system
    system = UnifiedStockIntelligence(config)
    system.run_system()
    
except Exception as e:
    print(f"❌ Failed to start application: {e}")
    print("💡 Make sure all dependencies are installed: pip install -r requirements.txt")
    input("Press Enter to exit...")
"@

Set-Content -Path "start_app.py" -Value $startScript
Write-Host "Created startup script: start_app.py" -ForegroundColor Green

# Display final instructions
Write-Host "`nSetup Complete!" -ForegroundColor Green
Write-Host "================" -ForegroundColor Green
Write-Host "To start the application, run:" -ForegroundColor White
Write-Host "  python start_app.py" -ForegroundColor Cyan
Write-Host "`nOr navigate to the backend directory and run your main script directly:" -ForegroundColor White
Write-Host "  cd D:\StockAI_Master\backend" -ForegroundColor Cyan
Write-Host "  python unified_stock_platform.py" -ForegroundColor Cyan

Write-Host "`nThe web dashboard will be available at: http://localhost:8000" -ForegroundColor Yellow

# Check if there are any existing log files
$logFiles = Get-ChildItem -Path "../logs" -Filter "*.log" -ErrorAction SilentlyContinue
if ($logFiles) {
    Write-Host "`nFound existing log files:" -ForegroundColor Yellow
    $logFiles | ForEach-Object { Write-Host "  $($_.Name) ($([math]::Round($_.Length/1KB,2)) KB)" -ForegroundColor Gray }
}