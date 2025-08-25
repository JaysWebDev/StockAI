# backend/auto_start.py
import os
import sys
import time
import subprocess
from datetime import datetime, time as dt_time
import threading

def run_daily_update():
    """Run the daily update at 4:00 AM"""
    while True:
        now = datetime.now()
        
        # Check if it's 4:00 AM
        if now.hour == 4 and now.minute == 0:
            print(f"\n🔄 Running daily update at {now.strftime('%Y-%m-%d %H:%M')}")
            try:
                # Run the daily scheduler
                subprocess.run([sys.executable, "../scheduler/daily_scheduler.py"], check=True)
                print("✅ Daily update completed successfully!")
            except subprocess.CalledProcessError as e:
                print(f"❌ Daily update failed: {e}")
            
            # Sleep for 1 hour to avoid multiple runs
            time.sleep(3600)
        else:
            # Sleep for 1 minute and check again
            time.sleep(60)

def start_server():
    """Start the FastAPI server"""
    print("🚀 Starting StockAI Server...")
    print("📊 Dashboard: http://localhost:8000")
    print("⏰ Daily auto-update: 4:00 AM")
    print("🔄 Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        # Start the FastAPI server
        subprocess.run([sys.executable, "main.py"])
    except KeyboardInterrupt:
        print("\n⏹️ Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")

if __name__ == "__main__":
    # First check dependencies
    print("🔍 Checking dependencies...")
    dependencies_ok = subprocess.call([sys.executable, "check_dependencies.py"])
    
    if dependencies_ok == 0:
        # Start daily update thread
        update_thread = threading.Thread(target=run_daily_update, daemon=True)
        update_thread.start()
        
        # Start the server
        start_server()
    else:
        print("❌ Failed to install required dependencies")
        input("Press Enter to exit...")