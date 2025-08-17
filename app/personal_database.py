"""
Personal Database Infrastructure

This module handles the SQLite database setup, migrations, encrypted storage,
and backup/recovery mechanisms for the personal assistant.
"""

import sqlite3
import json
import shutil
import gzip
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging
from contextlib import contextmanager
import threading
from dataclasses import asdict

logger = logging.getLogger(__name__)


class DatabaseMigration:
    """Represents a database migration"""
    
    def __init__(self, version: int, description: str, up_sql: str, down_sql: str = ""):
        self.version = version
        self.description = description
        self.up_sql = up_sql
        self.down_sql = down_sql


class PersonalDatabase:
    """Manages the personal assistant SQLite database with migrations and backups"""
    
    CURRENT_VERSION = 1
    
    def __init__(self, db_path: str = "personal_assistant.db", backup_dir: str = "backups"):
        self.db_path = db_path
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
        self._lock = threading.RLock()
        self._migrations = self._get_migrations()
        
        # Initialize database
        self._init_database()
        self._run_migrations()
    
    def _get_migrations(self) -> List[DatabaseMigration]:
        """Define database migrations"""
        return [
            DatabaseMigration(
                version=1,
                description="Initial schema setup",
                up_sql="""
                -- Core user contexts table
                CREATE TABLE IF NOT EXISTS user_contexts (
                    user_id TEXT PRIMARY KEY,
                    context_data TEXT NOT NULL,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    context_version INTEGER DEFAULT 1
                );
                
                -- User sessions table
                CREATE TABLE IF NOT EXISTS user_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    session_data TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT TRUE,
                    expires_at TIMESTAMP
                );
                
                -- Context history for tracking changes
                CREATE TABLE IF NOT EXISTS context_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    context_snapshot TEXT NOT NULL,
                    changes TEXT,
                    trigger_event TEXT,
                    metadata TEXT
                );
                
                -- User interactions table
                CREATE TABLE IF NOT EXISTS user_interactions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    interaction_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    response TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    context_data TEXT,
                    feedback_score REAL,
                    metadata TEXT
                );
                
                -- User permissions table
                CREATE TABLE IF NOT EXISTS user_permissions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    permission_type TEXT NOT NULL,
                    granted BOOLEAN DEFAULT FALSE,
                    granted_at TIMESTAMP,
                    expires_at TIMESTAMP,
                    scope TEXT,
                    revoked BOOLEAN DEFAULT FALSE,
                    revoked_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, permission_type)
                );
                
                -- User consent tracking
                CREATE TABLE IF NOT EXISTS user_consent (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    data_category TEXT NOT NULL,
                    consent_status TEXT NOT NULL,
                    granted_at TIMESTAMP,
                    expires_at TIMESTAMP,
                    revoked_at TIMESTAMP,
                    consent_details TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, data_category)
                );
                
                -- Encrypted data storage
                CREATE TABLE IF NOT EXISTS encrypted_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    data_key TEXT NOT NULL,
                    encrypted_content BLOB NOT NULL,
                    data_category TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, data_key)
                );
                
                -- Data deletion requests
                CREATE TABLE IF NOT EXISTS data_deletion_requests (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    data_categories TEXT NOT NULL,
                    requested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    processed_at TIMESTAMP,
                    status TEXT DEFAULT 'pending',
                    details TEXT
                );
                
                -- Audit log for security and compliance
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    resource TEXT,
                    details TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    ip_address TEXT,
                    user_agent TEXT
                );
                
                -- File system access log
                CREATE TABLE IF NOT EXISTS file_access_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    access_type TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    success BOOLEAN DEFAULT TRUE,
                    error_message TEXT
                );
                
                -- Personal knowledge base
                CREATE TABLE IF NOT EXISTS knowledge_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source_file TEXT,
                    category TEXT,
                    tags TEXT,
                    embedding BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                -- Task management
                CREATE TABLE IF NOT EXISTS user_tasks (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT,
                    status TEXT DEFAULT 'pending',
                    priority INTEGER DEFAULT 0,
                    due_date TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP,
                    project_id TEXT,
                    metadata TEXT
                );
                
                -- Project management
                CREATE TABLE IF NOT EXISTS user_projects (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT,
                    status TEXT DEFAULT 'active',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT
                );
                
                -- Database metadata and versioning
                CREATE TABLE IF NOT EXISTS database_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                -- Indexes for performance
                CREATE INDEX IF NOT EXISTS idx_user_interactions_user_timestamp 
                ON user_interactions(user_id, timestamp);
                
                CREATE INDEX IF NOT EXISTS idx_context_history_user_timestamp 
                ON context_history(user_id, timestamp);
                
                CREATE INDEX IF NOT EXISTS idx_audit_log_user_timestamp 
                ON audit_log(user_id, timestamp);
                
                CREATE INDEX IF NOT EXISTS idx_file_access_user_timestamp 
                ON file_access_log(user_id, timestamp);
                
                CREATE INDEX IF NOT EXISTS idx_knowledge_items_user_category 
                ON knowledge_items(user_id, category);
                
                CREATE INDEX IF NOT EXISTS idx_user_tasks_user_status 
                ON user_tasks(user_id, status);
                
                CREATE INDEX IF NOT EXISTS idx_user_sessions_user_active 
                ON user_sessions(user_id, is_active);
                """,
                down_sql="-- No rollback for initial schema"
            )
        ]
    
    def _init_database(self):
        """Initialize the database file and basic structure"""
        with sqlite3.connect(self.db_path) as conn:
            # Enable foreign keys
            conn.execute("PRAGMA foreign_keys = ON")
            
            # Set journal mode to WAL for better concurrency
            conn.execute("PRAGMA journal_mode = WAL")
            
            # Set synchronous mode for better performance
            conn.execute("PRAGMA synchronous = NORMAL")
            
            # Create migrations table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS database_migrations (
                    version INTEGER PRIMARY KEY,
                    description TEXT NOT NULL,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
    
    def _run_migrations(self):
        """Run pending database migrations"""
        with sqlite3.connect(self.db_path) as conn:
            # Get current version
            cursor = conn.execute("""
                SELECT MAX(version) FROM database_migrations
            """)
            current_version = cursor.fetchone()[0] or 0
            
            # Run pending migrations
            for migration in self._migrations:
                if migration.version > current_version:
                    logger.info(f"Running migration {migration.version}: {migration.description}")
                    
                    try:
                        # Execute migration SQL
                        conn.executescript(migration.up_sql)
                        
                        # Record migration
                        conn.execute("""
                            INSERT INTO database_migrations (version, description)
                            VALUES (?, ?)
                        """, (migration.version, migration.description))
                        
                        conn.commit()
                        logger.info(f"Migration {migration.version} completed successfully")
                        
                    except Exception as e:
                        logger.error(f"Migration {migration.version} failed: {e}")
                        conn.rollback()
                        raise
            
            # Update database version metadata
            conn.execute("""
                INSERT OR REPLACE INTO database_metadata (key, value)
                VALUES ('version', ?)
            """, (str(self.CURRENT_VERSION),))
            conn.commit()
    
    @contextmanager
    def get_connection(self):
        """Get a database connection with proper locking"""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # Enable column access by name
            try:
                yield conn
            finally:
                conn.close()
    
    def execute_query(self, query: str, params: Tuple = ()) -> List[sqlite3.Row]:
        """Execute a SELECT query and return results"""
        with self.get_connection() as conn:
            cursor = conn.execute(query, params)
            return cursor.fetchall()
    
    def execute_update(self, query: str, params: Tuple = ()) -> int:
        """Execute an INSERT/UPDATE/DELETE query and return affected rows"""
        with self.get_connection() as conn:
            cursor = conn.execute(query, params)
            conn.commit()
            return cursor.rowcount
    
    def execute_transaction(self, queries: List[Tuple[str, Tuple]]) -> bool:
        """Execute multiple queries in a transaction"""
        with self.get_connection() as conn:
            try:
                for query, params in queries:
                    conn.execute(query, params)
                conn.commit()
                return True
            except Exception as e:
                logger.error(f"Transaction failed: {e}")
                conn.rollback()
                return False
    
    def create_backup(self, backup_name: Optional[str] = None) -> str:
        """Create a compressed backup of the database"""
        if not backup_name:
            backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db.gz"
        
        backup_path = self.backup_dir / backup_name
        
        try:
            # Create compressed backup
            with open(self.db_path, 'rb') as f_in:
                with gzip.open(backup_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Record backup metadata
            with self.get_connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO database_metadata (key, value)
                    VALUES (?, ?)
                """, (f"backup_{backup_name}", json.dumps({
                    "created_at": datetime.now().isoformat(),
                    "size": backup_path.stat().st_size,
                    "original_size": Path(self.db_path).stat().st_size
                })))
                conn.commit()
            
            logger.info(f"Database backup created: {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            if backup_path.exists():
                backup_path.unlink()
            raise
    
    def restore_backup(self, backup_path: str) -> bool:
        """Restore database from a backup"""
        backup_file = Path(backup_path)
        if not backup_file.exists():
            logger.error(f"Backup file not found: {backup_path}")
            return False
        
        try:
            # Create a backup of current database before restore
            current_backup = self.create_backup(f"pre_restore_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db.gz")
            
            # Restore from backup
            temp_db = f"{self.db_path}.temp"
            
            with gzip.open(backup_file, 'rb') as f_in:
                with open(temp_db, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Verify restored database
            if self._verify_database_integrity(temp_db):
                # Replace current database
                shutil.move(temp_db, self.db_path)
                logger.info(f"Database restored from: {backup_path}")
                return True
            else:
                logger.error("Restored database failed integrity check")
                os.remove(temp_db)
                return False
                
        except Exception as e:
            logger.error(f"Failed to restore backup: {e}")
            return False
    
    def _verify_database_integrity(self, db_path: str) -> bool:
        """Verify database integrity"""
        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.execute("PRAGMA integrity_check")
                result = cursor.fetchone()[0]
                return result == "ok"
        except Exception as e:
            logger.error(f"Database integrity check failed: {e}")
            return False
    
    def cleanup_old_backups(self, keep_days: int = 30, keep_count: int = 10) -> None:
        """Clean up old backup files"""
        if not self.backup_dir.exists():
            return
        
        backup_files = []
        for backup_file in self.backup_dir.glob("backup_*.db.gz"):
            try:
                stat = backup_file.stat()
                backup_files.append((backup_file, stat.st_mtime))
            except OSError:
                continue
        
        # Sort by modification time (newest first)
        backup_files.sort(key=lambda x: x[1], reverse=True)
        
        cutoff_time = datetime.now().timestamp() - (keep_days * 24 * 3600)
        
        # Keep the most recent backups and those within the time limit
        for i, (backup_file, mtime) in enumerate(backup_files):
            if i >= keep_count and mtime < cutoff_time:
                try:
                    backup_file.unlink()
                    logger.info(f"Deleted old backup: {backup_file}")
                except OSError as e:
                    logger.error(f"Failed to delete backup {backup_file}: {e}")
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics and health information"""
        with self.get_connection() as conn:
            # Get table sizes
            tables = [
                'user_contexts', 'user_sessions', 'context_history', 'user_interactions',
                'user_permissions', 'user_consent', 'encrypted_data', 'audit_log',
                'file_access_log', 'knowledge_items', 'user_tasks', 'user_projects'
            ]
            
            table_stats = {}
            for table in tables:
                try:
                    cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    table_stats[table] = count
                except sqlite3.OperationalError:
                    table_stats[table] = 0
            
            # Get database size
            db_size = Path(self.db_path).stat().st_size
            
            # Get version info
            cursor = conn.execute("""
                SELECT value FROM database_metadata WHERE key = 'version'
            """)
            version_row = cursor.fetchone()
            version = version_row[0] if version_row else "unknown"
            
            # Get last backup info
            backup_files = list(self.backup_dir.glob("backup_*.db.gz"))
            last_backup = None
            if backup_files:
                latest_backup = max(backup_files, key=lambda x: x.stat().st_mtime)
                last_backup = {
                    "file": str(latest_backup),
                    "created": datetime.fromtimestamp(latest_backup.stat().st_mtime).isoformat(),
                    "size": latest_backup.stat().st_size
                }
            
            return {
                "database_size": db_size,
                "version": version,
                "table_counts": table_stats,
                "last_backup": last_backup,
                "backup_count": len(backup_files),
                "integrity_ok": self._verify_database_integrity(self.db_path)
            }
    
    def vacuum_database(self) -> bool:
        """Vacuum the database to reclaim space and optimize performance"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("VACUUM")
                conn.commit()
            
            logger.info("Database vacuum completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Database vacuum failed: {e}")
            return False
    
    def analyze_database(self) -> bool:
        """Analyze the database to update query planner statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("ANALYZE")
                conn.commit()
            
            logger.info("Database analysis completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Database analysis failed: {e}")
            return False
    
    def setup_automatic_maintenance(self) -> None:
        """Set up automatic database maintenance tasks"""
        # This would typically be called by a scheduler
        # For now, we'll just define the maintenance tasks
        
        def maintenance_task():
            try:
                # Clean up old backups
                self.cleanup_old_backups()
                
                # Create daily backup
                self.create_backup()
                
                # Vacuum database weekly (this would need scheduling logic)
                # self.vacuum_database()
                
                # Analyze database for query optimization
                self.analyze_database()
                
                logger.info("Automatic maintenance completed")
                
            except Exception as e:
                logger.error(f"Automatic maintenance failed: {e}")
        
        # In a real implementation, this would be scheduled
        # For now, just log that maintenance is available
        logger.info("Database maintenance tasks are available")
    
    def close(self):
        """Close database connections and cleanup"""
        # In this implementation, connections are managed per-operation
        # This method is here for interface completeness
        logger.info("Database manager closed")