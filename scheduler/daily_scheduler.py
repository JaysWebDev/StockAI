 #!/usr/bin/env python3
"""
Enhanced Daily Scheduler for Unified Stock Intelligence System
Integrates with the unified system for comprehensive daily updates
Now includes database maintenance and cleanup
"""

import sys
import os
import yaml
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
import shutil

# Add the current directory to path to import our unified system
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from unified_stock_intelligence import UnifiedStockIntelligence, Config

class DatabaseMaintenance:
    def __init__(self, db_path):
        self.db_path = db_path
        self.archive_dir = "../archive"
        self.retention_days = 90  # Keep 90 days of active data
        self.archive_days = 365   # Keep archives for 1 year
        
    def cleanup_old_data(self):
        """Remove data older than retention period"""
        cutoff_date = (datetime.now() - timedelta(days=self.retention_days)).strftime('%Y-%m-%d')
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Clean price history
            cursor.execute("DELETE FROM Price_History WHERE date < ?", (cutoff_date,))
            price_deleted = cursor.rowcount
            
            # Clean technical indicators
            cursor.execute("DELETE FROM Technical_Indicators WHERE date < ?", (cutoff_date,))
            tech_deleted = cursor.rowcount
            
            # Clean old AI decisions (keep only latest per company per day)
            cursor.execute("""
                DELETE FROM AI_Decisions 
                WHERE created_at < datetime('now', '-7 days')
                AND decision_id NOT IN (
                    SELECT MAX(decision_id) 
                    FROM AI_Decisions 
                    WHERE date(created_at) = date(created_at)
                    GROUP BY company_id, date(created_at)
                )
            """)
            ai_deleted = cursor.rowcount
            
            conn.commit()
            print(f"🗑️  Cleaned up: {price_deleted} price records, {tech_deleted} tech records, {ai_deleted} AI decisions")
            
        except Exception as e:
            print(f"❌ Error during cleanup: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def create_archive(self):
        """Create monthly archive of old data"""
        archive_date = datetime.now().strftime('%Y%m')
        archive_path = f"{self.archive_dir}/archive_{archive_date}.db"
        
        # Create archive directory if it doesn't exist
        os.makedirs(self.archive_dir, exist_ok=True)
        
        # Copy current database as archive
        shutil.copy2(self.db_path, archive_path)
        print(f"📦 Created archive: {archive_path}")
        
        # Clean up old archives
        self.cleanup_old_archives()
    
    def cleanup_old_archives(self):
        """Remove archives older than 1 year"""
        cutoff_date = datetime.now() - timedelta(days=self.archive_days)
        
        for archive_file in Path(self.archive_dir).glob("archive_*.db"):
            file_date_str = archive_file.stem.split('_')[1]
            try:
                file_date = datetime.strptime(file_date_str, '%Y%m')
                
                if file_date < cutoff_date:
                    archive_file.unlink()
                    print(f"🗑️  Removed old archive: {archive_file.name}")
            except ValueError:
                # Skip files with invalid date format
                continue
    
    def optimize_database(self):
        """Optimize database performance"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Vacuum to reclaim space
            cursor.execute("VACUUM")
            
            # Rebuild indexes
            cursor.execute("REINDEX")
            
            # Analyze for better query planning
            cursor.execute("ANALYZE")
            
            conn.commit()
            print("⚡ Database optimized successfully")
            
        except Exception as e:
            print(f"❌ Error optimizing database: {e}")
        finally:
            conn.close()

def load_config():
    """Load configuration from config.yaml with proper error handling"""
    config_path = Path("../config/config.yaml")
    
    if not config_path.exists():
        print("⚠️ config.yaml not found, using default configuration")
        return Config(), ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    
    try:
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Create basic config with only the parameters Config expects
        basic_config_params = {
            'alpha_vantage_api_key': config_data.get('alpha_vantage_api_key', 'DQ4I233CG66GOW9T'),
            'email': config_data.get('email', 'jaysweb@proton.me'),
            'db_path': config_data.get('db_path', '../data/unified_stock_intelligence.db'),
            'log_level': config_data.get('log_level', 'INFO'),
            'web_port': config_data.get('web_port', 8000),
            'rate_limit_delay': config_data.get('rate_limit_delay', 12),
            'max_retries': config_data.get('max_retries', 3)
        }
        
        # Create the config object
        config = Config(**basic_config_params)
        
        # Get the watchlist separately
        watchlist = config_data.get('default_watchlist', [
            "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", 
            "NVDA", "META", "JPM", "V", "JNJ"
        ])
        
        print(f"✅ Configuration loaded successfully")
        print(f"📋 Watchlist contains {len(watchlist)} securities")
        
        return config, watchlist
        
    except yaml.YAMLError as e:
        print(f"❌ Error parsing config.yaml: {e}")
        print("Using default configuration instead")
        return Config(), ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    except Exception as e:
        print(f"❌ Unexpected error loading config: {e}")
        print("Using default configuration instead")
        return Config(), ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

def run_database_maintenance(config):
    """Run database maintenance routines"""
    print("\n" + "=" * 60)
    print("🗄️  DATABASE MAINTENANCE")
    print("=" * 60)
    
    try:
        maintenance = DatabaseMaintenance(config.db_path)
        maintenance.cleanup_old_data()
        maintenance.create_archive()
        maintenance.optimize_database()
        print("✅ Database maintenance completed successfully")
        return True
    except Exception as e:
        print(f"❌ Database maintenance failed: {e}")
        return False

def main():
    """Run comprehensive daily updates"""
    print(f"🚀 Starting daily update at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    try:
        # Load configuration and watchlist
        print("📊 Loading configuration...")
        config, watchlist_tickers = load_config()
        
        # Initialize the unified system
        print("🔧 Initializing Unified Stock Intelligence System...")
        system = UnifiedStockIntelligence(config)
        
        print(f"📋 Processing {len(watchlist_tickers)} stocks in watchlist:")
        print(f"   {', '.join(watchlist_tickers[:10])}{'...' if len(watchlist_tickers) > 10 else ''}")
        print()
        
        # Run batch analysis
        print("🔄 Starting batch analysis...")
        results = system.run_batch_analysis(watchlist_tickers)
        
        # Generate summary statistics
        successful_analyses = len([r for r in results.values() if 'error' not in r])
        failed_analyses = len([r for r in results.values() if 'error' in r])
        
        print("\n" + "=" * 60)
        print("📈 DAILY UPDATE SUMMARY")
        print("=" * 60)
        print(f"✅ Successful analyses: {successful_analyses}")
        print(f"❌ Failed analyses: {failed_analyses}")
        print(f"📊 Total processed: {len(results)}")
        
        # Show key decisions
        print("\n🎯 KEY AI DECISIONS:")
        print("-" * 40)
        
        buy_signals = []
        sell_signals = []
        hold_signals = []
        
        for ticker, result in results.items():
            if 'error' not in result:
                decision = result.get('decision', 'N/A')
                confidence = result.get('confidence', 0)
                score = result.get('ai_score', 0)
                
                signal_text = f"{ticker}: {decision} (Score: {score}, Confidence: {confidence*100:.0f}%)"
                
                if 'BUY' in decision:
                    buy_signals.append(signal_text)
                elif 'SELL' in decision:
                    sell_signals.append(signal_text)
                elif 'HOLD' in decision:
                    hold_signals.append(signal_text)
        
        if buy_signals:
            print("🟢 BUY SIGNALS:")
            for signal in buy_signals[:5]:  # Show top 5 only
                print(f"  • {signal}")
            if len(buy_signals) > 5:
                print(f"  • ... and {len(buy_signals) - 5} more")
        
        if sell_signals:
            print("🔴 SELL SIGNALS:")
            for signal in sell_signals[:5]:
                print(f"  • {signal}")
            if len(sell_signals) > 5:
                print(f"  • ... and {len(sell_signals) - 5} more")
        
        if hold_signals:
            print("🟡 HOLD SIGNALS:")
            for signal in hold_signals[:5]:
                print(f"  • {signal}")
            if len(hold_signals) > 5:
                print(f"  • ... and {len(hold_signals) - 5} more")
        
        if not buy_signals and not sell_signals and not hold_signals:
            print("📊 No clear signals detected")
        
        # Show any errors
        if failed_analyses > 0:
            print(f"\n❌ FAILED ANALYSES ({failed_analyses}):")
            print("-" * 40)
            error_count = 0
            for ticker, result in results.items():
                if 'error' in result and error_count < 5:
                    print(f"  • {ticker}: {result['error'][:50]}...")
                    error_count += 1
            if failed_analyses > 5:
                print(f"  • ... and {failed_analyses - 5} more errors")
        
        # Run database maintenance
        maintenance_success = run_database_maintenance(config)
        
        # Save daily report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = f"../reports/daily_report_{timestamp}.txt"
        
        # Create reports directory if it doesn't exist
        os.makedirs("../reports", exist_ok=True)
        
        with open(report_file, 'w') as f:
            f.write(f"Daily Stock Intelligence Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Watchlist: {', '.join(watchlist_tickers)}\n\n")
            
            f.write(f"Summary:\n")
            f.write(f"- Successful analyses: {successful_analyses}\n")
            f.write(f"- Failed analyses: {failed_analyses}\n")
            f.write(f"- Total processed: {len(results)}\n")
            f.write(f"- Database maintenance: {'Success' if maintenance_success else 'Failed'}\n\n")
            
            f.write("Detailed Results:\n")
            f.write("-" * 40 + "\n")
            
            for ticker, result in results.items():
                if 'error' not in result:
                    f.write(f"{ticker}: {result.get('decision', 'N/A')} ")
                    f.write(f"(Score: {result.get('ai_score', 0)}, ")
                    f.write(f"Confidence: {result.get('confidence', 0)*100:.0f}%)\n")
                else:
                    f.write(f"{ticker}: ERROR - {result['error']}\n")
        
        print(f"\n📄 Daily report saved: {report_file}")
        
        # Get dashboard data for final summary
        try:
            dashboard_data = system.get_dashboard_data()
            total_decisions = dashboard_data['stats']['total_decisions']
            avg_confidence = dashboard_data['stats']['avg_confidence']
            
            print(f"\n📊 SYSTEM STATISTICS:")
            print(f"  • Total decisions in database: {total_decisions}")
            print(f"  • Average AI confidence: {avg_confidence*100:.1f}%")
        except Exception as e:
            print(f"⚠️ Could not retrieve system statistics: {e}")
        
        print("\n✅ Daily update completed successfully!")
        
        # Optional: Check for alerts
        print("\n🔔 CHECKING FOR ALERTS...")
        alert_count = 0
        
        for ticker in watchlist_tickers[:10]:  # Check first 10 only to save time
            try:
                alerts = system.check_alerts(ticker)
                if alerts:
                    print(f"⚠️ {ticker}: {len(alerts)} alert(s)")
                    for alert in alerts[:2]:  # Show first 2 alerts per ticker
                        print(f"   • {alert['message']}")
                    alert_count += len(alerts)
            except Exception as e:
                print(f"⚠️ Error checking alerts for {ticker}: {e}")
        
        if alert_count == 0:
            print("✅ No alerts triggered")
        else:
            print(f"⚠️ Total alerts: {alert_count}")
            
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR in daily update: {e}")
        print("💡 Check your configuration and API keys")
        
        # Log the error
        os.makedirs("../logs", exist_ok=True)
        error_log = f"../logs/error_log_{datetime.now().strftime('%Y%m%d')}.txt"
        with open(error_log, 'a') as f:
            f.write(f"{datetime.now()}: {str(e)}\n")
        
        return 1  # Exit with error code
    
    print(f"\n🏁 Daily update finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return 0  # Success

if __name__ == "__main__":
    exit_code = main()
    
    # If running manually, pause so user can see results
    if len(sys.argv) == 1:  # No command line arguments = manual run
        input("\nPress Enter to exit...")
    
    sys.exit(exit_code)