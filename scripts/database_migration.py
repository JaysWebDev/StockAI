#!/usr/bin/env python3
"""
Database Migration Scripts for StockAI
Safely create and manage database schema
"""

import sqlite3
import os
import sys
from datetime import datetime
from pathlib import Path
import shutil

class DatabaseMigration:
    """Handle database migrations and setup"""

    def __init__(self, db_path: str = "unified_stock_intelligence.db"):
        self.db_path = db_path
        self.backup_dir = Path("backups")
        self.backup_dir.mkdir(exist_ok=True)

    def create_backup(self) -> str:
        """Create a backup of the existing database"""
        if not os.path.exists(self.db_path):
            print(f"⚠️ Database {self.db_path} does not exist, skipping backup")
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"database_backup_{timestamp}.db"
        backup_path = self.backup_dir / backup_name

        try:
            shutil.copy2(self.db_path, backup_path)
            print(f"✅ Database backed up to {backup_path}")
            return str(backup_path)
        except Exception as e:
            print(f"❌ Failed to create backup: {e}")
            return None

    def create_fresh_database(self):
        """Create a fresh database with all tables"""
        print("🗄️ Creating fresh database schema...")

        # Backup existing database if it exists
        backup_path = self.create_backup()

        # Connect to database (creates if doesn't exist)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Companies table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS Companies (
                    company_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT UNIQUE NOT NULL,
                    name TEXT,
                    exchange TEXT,
                    sector TEXT,
                    market_cap REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Price History table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS Price_History (
                    price_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    company_id INTEGER NOT NULL,
                    date DATE NOT NULL,
                    open_price REAL,
                    high_price REAL,
                    low_price REAL,
                    close_price REAL,
                    adj_close_price REAL,
                    volume INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (company_id) REFERENCES Companies(company_id),
                    UNIQUE(company_id, date)
                )
            ''')

            # Technical Indicators table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS Technical_Indicators (
                    indicator_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    company_id INTEGER NOT NULL,
                    date DATE NOT NULL,
                    sma_20 REAL,
                    sma_50 REAL,
                    sma_200 REAL,
                    ema_12 REAL,
                    ema_26 REAL,
                    rsi_14 REAL,
                    macd REAL,
                    macd_signal REAL,
                    macd_histogram REAL,
                    bb_upper REAL,
                    bb_middle REAL,
                    bb_lower REAL,
                    stoch_k REAL,
                    stoch_d REAL,
                    williams_r REAL,
                    atr_14 REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (company_id) REFERENCES Companies(company_id),
                    UNIQUE(company_id, date)
                )
            ''')

            # AI Decisions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS AI_Decisions (
                    decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    company_id INTEGER NOT NULL,
                    decision TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    reasoning TEXT,
                    price_at_decision REAL,
                    target_price REAL,
                    stop_loss REAL,
                    risk_level TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (company_id) REFERENCES Companies(company_id)
                )
            ''')

            # Scanner Signals table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS Scanner_Signals (
                    signal_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    company_id INTEGER NOT NULL,
                    scanner_type TEXT NOT NULL,
                    signal_strength REAL,
                    signal_date DATE NOT NULL,
                    signal_data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (company_id) REFERENCES Companies(company_id)
                )
            ''')

            # Portfolio Positions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS Portfolio_Positions (
                    position_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    company_id INTEGER NOT NULL,
                    entry_date DATE NOT NULL,
                    entry_price REAL NOT NULL,
                    quantity INTEGER NOT NULL,
                    position_type TEXT DEFAULT 'LONG',
                    exit_date DATE,
                    exit_price REAL,
                    realized_pnl REAL,
                    status TEXT DEFAULT 'OPEN',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (company_id) REFERENCES Companies(company_id)
                )
            ''')

            # System Metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS System_Metrics (
                    metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    metric_value REAL,
                    metric_date DATE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(metric_name, metric_date)
                )
            ''')

            # Create indexes for better performance
            self._create_indexes(cursor)

            conn.commit()
            print("✅ Database schema created successfully")

            # Insert sample data if this is a fresh database
            if backup_path is None:
                self._insert_sample_data(cursor)
                conn.commit()
                print("✅ Sample data inserted")

        except Exception as e:
            print(f"❌ Error creating database: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()

    def _create_indexes(self, cursor):
        """Create database indexes for performance"""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_companies_symbol ON Companies(symbol)",
            "CREATE INDEX IF NOT EXISTS idx_price_history_date ON Price_History(date)",
            "CREATE INDEX IF NOT EXISTS idx_price_history_company_date ON Price_History(company_id, date)",
            "CREATE INDEX IF NOT EXISTS idx_technical_indicators_date ON Technical_Indicators(date)",
            "CREATE INDEX IF NOT EXISTS idx_ai_decisions_date ON AI_Decisions(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_scanner_signals_date ON Scanner_Signals(signal_date)",
            "CREATE INDEX IF NOT EXISTS idx_portfolio_positions_date ON Portfolio_Positions(entry_date)",
        ]

        for index_sql in indexes:
            try:
                cursor.execute(index_sql)
            except Exception as e:
                print(f"⚠️ Warning: Could not create index: {e}")

    def _insert_sample_data(self, cursor):
        """Insert sample data for testing"""
        sample_companies = [
            ('AAPL', 'Apple Inc.', 'NASDAQ', 'Technology'),
            ('MSFT', 'Microsoft Corporation', 'NASDAQ', 'Technology'),
            ('GOOGL', 'Alphabet Inc.', 'NASDAQ', 'Technology'),
            ('AMZN', 'Amazon.com Inc.', 'NASDAQ', 'Consumer Discretionary'),
            ('TSLA', 'Tesla Inc.', 'NASDAQ', 'Consumer Discretionary'),
        ]

        cursor.executemany('''
            INSERT OR IGNORE INTO Companies (symbol, name, exchange, sector)
            VALUES (?, ?, ?, ?)
        ''', sample_companies)

        print("✅ Sample companies added")

    def verify_database(self) -> bool:
        """Verify database integrity and structure"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Check if all required tables exist
            required_tables = [
                'Companies', 'Price_History', 'Technical_Indicators',
                'AI_Decisions', 'Scanner_Signals', 'Portfolio_Positions',
                'System_Metrics'
            ]

            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            existing_tables = [row[0] for row in cursor.fetchall()]

            missing_tables = []
            for table in required_tables:
                if table not in existing_tables:
                    missing_tables.append(table)

            if missing_tables:
                print(f"❌ Missing tables: {missing_tables}")
                return False

            # Check table row counts
            for table in required_tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                print(f"✅ {table}: {count} records")

            conn.close()
            print("✅ Database verification complete")
            return True

        except Exception as e:
            print(f"❌ Database verification failed: {e}")
            return False

    def upgrade_schema(self):
        """Upgrade database schema for new features"""
        print("🔄 Checking for schema upgrades...")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Add new columns if they don't exist
            upgrades = [
                ("Companies", "market_cap", "REAL"),
                ("Companies", "updated_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"),
                ("Price_History", "adj_close_price", "REAL"),
                ("AI_Decisions", "risk_level", "TEXT"),
                ("Portfolio_Positions", "position_type", "TEXT DEFAULT 'LONG'"),
            ]

            for table, column, column_type in upgrades:
                try:
                    cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")
                    print(f"✅ Added {column} to {table}")
                except sqlite3.OperationalError as e:
                    if "duplicate column name" in str(e):
                        print(f"⚠️ Column {column} already exists in {table}")
                    else:
                        print(f"❌ Failed to add {column} to {table}: {e}")

            conn.commit()
            print("✅ Schema upgrade complete")

        except Exception as e:
            print(f"❌ Schema upgrade failed: {e}")
            conn.rollback()
        finally:
            conn.close()

    def export_schema(self, output_file: str = "schema.sql"):
        """Export current database schema"""
        try:
            conn = sqlite3.connect(self.db_path)

            with open(output_file, 'w') as f:
                for line in conn.iterdump():
                    # Only export schema, not data
                    if line.startswith('CREATE'):
                        f.write(line + '\n')

            print(f"✅ Schema exported to {output_file}")
            conn.close()

        except Exception as e:
            print(f"❌ Schema export failed: {e}")

def main():
    """Main migration script"""
    print("🗄️ StockAI Database Migration Tool")
    print("=" * 40)

    # Get database path from environment or use default
    db_path = os.getenv('DATABASE_PATH', 'unified_stock_intelligence.db')
    print(f"📍 Database: {db_path}")

    migration = DatabaseMigration(db_path)

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "create":
            migration.create_fresh_database()
        elif command == "verify":
            migration.verify_database()
        elif command == "upgrade":
            migration.upgrade_schema()
        elif command == "export":
            output_file = sys.argv[2] if len(sys.argv) > 2 else "schema.sql"
            migration.export_schema(output_file)
        elif command == "backup":
            migration.create_backup()
        else:
            print(f"❌ Unknown command: {command}")
            print("Available commands: create, verify, upgrade, export, backup")
    else:
        # Interactive mode
        print("\n📋 Available operations:")
        print("1. Create fresh database")
        print("2. Verify database")
        print("3. Upgrade schema")
        print("4. Export schema")
        print("5. Create backup")

        try:
            choice = input("\nSelect operation (1-5): ").strip()

            if choice == "1":
                migration.create_fresh_database()
            elif choice == "2":
                migration.verify_database()
            elif choice == "3":
                migration.upgrade_schema()
            elif choice == "4":
                migration.export_schema()
            elif choice == "5":
                migration.create_backup()
            else:
                print("❌ Invalid choice")
        except KeyboardInterrupt:
            print("\n👋 Migration cancelled")

if __name__ == "__main__":
    main()