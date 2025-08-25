REM FILE 4: quick_analysis.bat
@echo off
REM quick_analysis.bat - Analyze a single stock quickly
echo 🎯 Quick Stock Analysis Tool
echo.

set /p ticker="Enter stock ticker (e.g. AAPL): "

if "%ticker%"=="" (
    echo ❌ No ticker entered!
    pause
    exit /b 1
)

echo.
echo 🧠 Analyzing %ticker%...
echo.

python -c "
from unified_stock_intelligence import UnifiedStockIntelligence, Config
import sys

try:
    system = UnifiedStockIntelligence(Config())
    result = system.run_ai_analysis('%ticker%')
    
    if 'error' in result:
        print(f'❌ Error: {result[\"error\"]}')
    else:
        print(f'🎯 {result[\"ticker\"]}: {result[\"decision\"]}')
        print(f'📊 AI Score: {result[\"ai_score\"]}/100')
        print(f'🎲 Confidence: {result[\"confidence\"]*100:.0f}%%')
        print(f'💰 Expected Return: {result[\"expected_return\"]*100:+.1f}%%')
        print(f'⚠️ Risk Level: {result[\"risk_level\"]}')
        print()
        print('💡 Reasoning:')
        print(result[\"reasoning\"])
        
except Exception as e:
    print(f'❌ Analysis failed: {e}')
    sys.exit(1)
"

echo.
pause