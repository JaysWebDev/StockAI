# backend/main.py
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
import sys
import os
from pathlib import Path
import sqlite3
import pandas as pd
from datetime import datetime, timedelta

# Add the backend directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from fastapi.templating import Jinja2Templates
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False
    print("⚠️ Jinja2 not installed. Using basic HTML serving.")

from unified_stock_intelligence import UnifiedStockIntelligence, Config

app = FastAPI(title="StockAI Master API", version="1.0.0")

# Serve static files from frontend directory
app.mount("/static", StaticFiles(directory="../frontend"), name="static")

# Configure templates if Jinja2 is available
if JINJA2_AVAILABLE:
    templates = Jinja2Templates(directory="../frontend")
else:
    print("⚠️ Template functionality limited. Install jinja2 for full features.")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the stock intelligence system
config = Config.load_from_file("../config/config.yaml")
stock_system = UnifiedStockIntelligence(config)

def serve_html_file(filename: str):
    """Serve HTML file directly without Jinja2 templates"""
    file_path = Path("../frontend") / filename
    if file_path.exists():
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return HTMLResponse(content)
    return HTMLResponse("<h1>File not found</h1>", status_code=404)

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Serve the main dashboard"""
    if JINJA2_AVAILABLE:
        return templates.TemplateResponse("dashboard.html", {"request": request})
    else:
        return serve_html_file("dashboard.html")

@app.get("/api/analyze/{ticker}")
async def analyze_stock(ticker: str):
    """Analyze a stock ticker"""
    try:
        result = stock_system.run_ai_analysis(ticker)
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/dashboard")
async def get_dashboard():
    """Get dashboard data"""
    try:
        data = stock_system.get_dashboard_data()
        return JSONResponse(data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stocks")
async def get_stocks(symbol: str = "AAPL", days: int = 30):
    """Get stock data for the dashboard"""
    try:
        # Get company ID
        company_id = stock_system.get_or_create_company(symbol, fetch_info=False)
        
        # Connect to database
        conn = sqlite3.connect(config.db_path)
        
        # Get price history
        query = """
            SELECT date, open_price as open, high_price as high, 
                   low_price as low, close_price as close, volume
            FROM Price_History 
            WHERE company_id = ? 
            ORDER BY date DESC 
            LIMIT ?
        """
        
        df = pd.read_sql_query(query, conn, params=(company_id, days))
        conn.close()
        
        if df.empty:
            return JSONResponse([])
        
        # Convert to list of dicts for JSON response
        result = df.to_dict('records')
        return JSONResponse(result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats")
async def get_stats():
    """Get system statistics"""
    try:
        conn = sqlite3.connect(config.db_path)
        
        stats = {
            "total_companies": pd.read_sql_query("SELECT COUNT(*) FROM Companies", conn).iloc[0,0],
            "total_decisions": pd.read_sql_query("SELECT COUNT(*) FROM AI_Decisions", conn).iloc[0,0],
            "buy_signals": pd.read_sql_query("SELECT COUNT(*) FROM AI_Decisions WHERE decision LIKE '%BUY%'", conn).iloc[0,0],
            "avg_confidence": pd.read_sql_query("SELECT AVG(confidence) FROM AI_Decisions", conn).iloc[0,0] or 0,
            "api_usage_today": 0,
            "last_update": datetime.now().isoformat()
        }
        
        conn.close()
        return JSONResponse(stats)
        
    except Exception as e:
        return JSONResponse({
            "total_companies": 0,
            "total_decisions": 0,
            "buy_signals": 0,
            "avg_confidence": 0,
            "api_usage_today": 0,
            "last_update": datetime.now().isoformat()
        })

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/api/install-status")
async def install_status():
    """Check if required packages are installed"""
    return {
        "jinja2_installed": JINJA2_AVAILABLE,
        "fastapi_installed": True,
        "uvicorn_installed": True
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting StockAI Server...")
    print("📊 Dashboard: http://localhost:8000")
    print("🔧 Install status:")
    print(f"   • Jinja2: {'✅' if JINJA2_AVAILABLE else '❌'}")
    print("   • FastAPI: ✅")
    print("   • Uvicorn: ✅")
    
    if not JINJA2_AVAILABLE:
        print("\n⚠️  Jinja2 not installed. Run 'install_dependencies.bat' for full features.")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)


@app.post("/api/refresh")
async def refresh_data():
    """Manual data refresh endpoint"""
    try:
        # Run the daily scheduler manually
        import subprocess
        result = subprocess.run([
            sys.executable, "../scheduler/daily_scheduler.py"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            return {"status": "success", "message": "Data refresh completed"}
        else:
            return {"status": "error", "message": result.stderr}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))