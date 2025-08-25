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
    from backend.unified_stock_intelligence import UnifiedStockIntelligence, Config
    
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
