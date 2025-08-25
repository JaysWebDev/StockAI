# scripts/database_maintenance.py
import sqlite3
import os
from datetime import datetime, timedelta
from pathlib import Path
import shutil

class DatabaseMaintenance:
    def __init__(self, db_path="../data/unified_stock_intelligence.db"):
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
            
            # Clean old AI decisions (keep only latest per company)
            cursor.execute("""
                DELETE FROM AI_Decisions 
                WHERE created_at < (
                    SELECT MAX(created_at) 
                    FROM AI_Decisions ad2 
                    WHERE ad2.company_id = AI_Decisions.company_id
                ) AND created_at < datetime('now', '-7 days')
            """)
            ai_deleted = cursor.rowcount
            
            conn.commit()
            print(f"Cleaned up: {price_deleted} price records, {tech_deleted} tech records, {ai_deleted} AI decisions")
            
        except Exception as e:
            print(f"Error during cleanup: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def create_archive(self):
        """Create monthly archive of old data"""
        archive_date = datetime.now().strftime('%Y%m')
        archive_path = f"{self.archive_dir}/archive_{archive_date}.db"
        
        # Copy current database as archive
        shutil.copy2(self.db_path, archive_path)
        print(f"Created archive: {archive_path}")
        
        # Clean up old archives
        self.cleanup_old_archives()
    
    def cleanup_old_archives(self):
        """Remove archives older than 1 year"""
        cutoff_date = datetime.now() - timedelta(days=self.archive_days)
        
        for archive_file in Path(self.archive_dir).glob("archive_*.db"):
            file_date_str = archive_file.stem.split('_')[1]
            file_date = datetime.strptime(file_date_str, '%Y%m')
            
            if file_date < cutoff_date:
                archive_file.unlink()
                print(f"Removed old archive: {archive_file.name}")
    
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
            print("Database optimized successfully")
            
        except Exception as e:
            print(f"Error optimizing database: {e}")
        finally:
            conn.close()

if __name__ == "__main__":
    maintenance = DatabaseMaintenance()
    maintenance.cleanup_old_data()
    maintenance.create_archive()
    maintenance.optimize_database()