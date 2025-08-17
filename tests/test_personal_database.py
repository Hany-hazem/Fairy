"""
Tests for Personal Database Infrastructure
"""

import pytest
import tempfile
import os
import sqlite3
import gzip
import json
from pathlib import Path
from datetime import datetime

from app.personal_database import PersonalDatabase, DatabaseMigration


class TestPersonalDatabase:
    """Test cases for PersonalDatabase"""
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing"""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_file:
            db_path = tmp_file.name
        
        try:
            yield db_path
        finally:
            # Cleanup
            if os.path.exists(db_path):
                os.unlink(db_path)
            # Also cleanup WAL files
            for ext in ["-wal", "-shm"]:
                wal_file = db_path + ext
                if os.path.exists(wal_file):
                    os.unlink(wal_file)
    
    @pytest.fixture
    def temp_backup_dir(self):
        """Create a temporary backup directory"""
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    def test_database_initialization(self, temp_db):
        """Test database initialization"""
        db = PersonalDatabase(temp_db)
        
        # Verify database file exists
        assert os.path.exists(temp_db)
        
        # Verify basic tables exist
        with sqlite3.connect(temp_db) as conn:
            cursor = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
            """)
            tables = [row[0] for row in cursor.fetchall()]
            
            expected_tables = [
                'user_contexts', 'user_sessions', 'context_history',
                'user_interactions', 'user_permissions', 'user_consent',
                'encrypted_data', 'data_deletion_requests', 'audit_log',
                'file_access_log', 'knowledge_items', 'user_tasks',
                'user_projects', 'database_metadata', 'database_migrations'
            ]
            
            for table in expected_tables:
                assert table in tables
    
    def test_database_migrations(self, temp_db):
        """Test database migration system"""
        db = PersonalDatabase(temp_db)
        
        # Verify migration was recorded
        with sqlite3.connect(temp_db) as conn:
            cursor = conn.execute("SELECT version, description FROM database_migrations")
            migrations = cursor.fetchall()
            
            assert len(migrations) >= 1
            assert migrations[0][0] == 1  # Version 1
            assert "Initial schema setup" in migrations[0][1]
    
    def test_get_connection(self, temp_db):
        """Test database connection management"""
        db = PersonalDatabase(temp_db)
        
        with db.get_connection() as conn:
            # Test that connection works
            cursor = conn.execute("SELECT 1")
            result = cursor.fetchone()
            assert result[0] == 1
            
            # Test row factory is set
            cursor = conn.execute("SELECT 1 as test_column")
            row = cursor.fetchone()
            assert row['test_column'] == 1
    
    def test_execute_query(self, temp_db):
        """Test executing SELECT queries"""
        db = PersonalDatabase(temp_db)
        
        # Insert test data
        with db.get_connection() as conn:
            conn.execute("""
                INSERT INTO user_contexts (user_id, context_data)
                VALUES (?, ?)
            """, ("test_user", '{"test": "data"}'))
            conn.commit()
        
        # Test query execution
        results = db.execute_query(
            "SELECT user_id, context_data FROM user_contexts WHERE user_id = ?",
            ("test_user",)
        )
        
        assert len(results) == 1
        assert results[0]['user_id'] == "test_user"
        assert results[0]['context_data'] == '{"test": "data"}'
    
    def test_execute_update(self, temp_db):
        """Test executing INSERT/UPDATE/DELETE queries"""
        db = PersonalDatabase(temp_db)
        
        # Test INSERT
        affected_rows = db.execute_update("""
            INSERT INTO user_contexts (user_id, context_data)
            VALUES (?, ?)
        """, ("test_user", '{"test": "data"}'))
        
        assert affected_rows == 1
        
        # Test UPDATE
        affected_rows = db.execute_update("""
            UPDATE user_contexts 
            SET context_data = ? 
            WHERE user_id = ?
        """, ('{"test": "updated"}', "test_user"))
        
        assert affected_rows == 1
        
        # Test DELETE
        affected_rows = db.execute_update("""
            DELETE FROM user_contexts WHERE user_id = ?
        """, ("test_user",))
        
        assert affected_rows == 1
    
    def test_execute_transaction(self, temp_db):
        """Test executing multiple queries in a transaction"""
        db = PersonalDatabase(temp_db)
        
        queries = [
            ("INSERT INTO user_contexts (user_id, context_data) VALUES (?, ?)", 
             ("user1", '{"data": "1"}')),
            ("INSERT INTO user_contexts (user_id, context_data) VALUES (?, ?)", 
             ("user2", '{"data": "2"}')),
            ("INSERT INTO user_sessions (session_id, user_id, session_data) VALUES (?, ?, ?)", 
             ("session1", "user1", '{"active": true}'))
        ]
        
        success = db.execute_transaction(queries)
        assert success is True
        
        # Verify all data was inserted
        results = db.execute_query("SELECT COUNT(*) as count FROM user_contexts")
        assert results[0]['count'] == 2
        
        results = db.execute_query("SELECT COUNT(*) as count FROM user_sessions")
        assert results[0]['count'] == 1
    
    def test_execute_transaction_rollback(self, temp_db):
        """Test transaction rollback on error"""
        db = PersonalDatabase(temp_db)
        
        # Insert valid data first
        db.execute_update("""
            INSERT INTO user_contexts (user_id, context_data)
            VALUES (?, ?)
        """, ("existing_user", '{"test": "data"}'))
        
        # Try transaction with duplicate key (should fail)
        queries = [
            ("INSERT INTO user_contexts (user_id, context_data) VALUES (?, ?)", 
             ("new_user", '{"data": "1"}')),
            ("INSERT INTO user_contexts (user_id, context_data) VALUES (?, ?)", 
             ("existing_user", '{"data": "2"}'))  # This should fail due to PRIMARY KEY
        ]
        
        success = db.execute_transaction(queries)
        assert success is False
        
        # Verify rollback - new_user should not exist
        results = db.execute_query("SELECT COUNT(*) as count FROM user_contexts WHERE user_id = ?", ("new_user",))
        assert results[0]['count'] == 0
    
    def test_create_backup(self, temp_db, temp_backup_dir):
        """Test creating database backup"""
        db = PersonalDatabase(temp_db, temp_backup_dir)
        
        # Add some data
        db.execute_update("""
            INSERT INTO user_contexts (user_id, context_data)
            VALUES (?, ?)
        """, ("backup_user", '{"test": "backup_data"}'))
        
        # Create backup
        backup_path = db.create_backup()
        
        assert os.path.exists(backup_path)
        assert backup_path.endswith('.db.gz')
        
        # Verify backup contains data
        with gzip.open(backup_path, 'rb') as f:
            backup_data = f.read()
            assert len(backup_data) > 0
    
    def test_restore_backup(self, temp_db, temp_backup_dir):
        """Test restoring from backup"""
        db = PersonalDatabase(temp_db, temp_backup_dir)
        
        # Add initial data
        db.execute_update("""
            INSERT INTO user_contexts (user_id, context_data)
            VALUES (?, ?)
        """, ("original_user", '{"test": "original_data"}'))
        
        # Create backup
        backup_path = db.create_backup("test_backup.db.gz")
        
        # Add more data after backup
        db.execute_update("""
            INSERT INTO user_contexts (user_id, context_data)
            VALUES (?, ?)
        """, ("new_user", '{"test": "new_data"}'))
        
        # Verify both users exist
        results = db.execute_query("SELECT COUNT(*) as count FROM user_contexts")
        assert results[0]['count'] == 2
        
        # Restore from backup
        success = db.restore_backup(backup_path)
        assert success is True
        
        # Verify only original user exists after restore
        results = db.execute_query("SELECT user_id FROM user_contexts")
        assert len(results) == 1
        assert results[0]['user_id'] == "original_user"
    
    def test_restore_nonexistent_backup(self, temp_db, temp_backup_dir):
        """Test restoring from nonexistent backup"""
        db = PersonalDatabase(temp_db, temp_backup_dir)
        
        success = db.restore_backup("nonexistent_backup.db.gz")
        assert success is False
    
    def test_cleanup_old_backups(self, temp_db, temp_backup_dir):
        """Test cleaning up old backup files"""
        db = PersonalDatabase(temp_db, temp_backup_dir)
        
        # Create multiple backups
        backup_paths = []
        for i in range(5):
            backup_path = db.create_backup(f"backup_{i}.db.gz")
            backup_paths.append(backup_path)
        
        # Verify all backups exist
        for backup_path in backup_paths:
            assert os.path.exists(backup_path)
        
        # Clean up old backups (keep only 3)
        db.cleanup_old_backups(keep_days=0, keep_count=3)
        
        # Count remaining backups
        remaining_backups = list(Path(temp_backup_dir).glob("backup_*.db.gz"))
        assert len(remaining_backups) == 3
    
    def test_get_database_stats(self, temp_db, temp_backup_dir):
        """Test getting database statistics"""
        db = PersonalDatabase(temp_db, temp_backup_dir)
        
        # Add some test data
        db.execute_update("""
            INSERT INTO user_contexts (user_id, context_data)
            VALUES (?, ?)
        """, ("stats_user", '{"test": "stats_data"}'))
        
        db.execute_update("""
            INSERT INTO user_interactions (id, user_id, interaction_type, content)
            VALUES (?, ?, ?, ?)
        """, ("interaction1", "stats_user", "query", "test query"))
        
        # Create a backup
        db.create_backup()
        
        # Get stats
        stats = db.get_database_stats()
        
        assert "database_size" in stats
        assert "version" in stats
        assert "table_counts" in stats
        assert "last_backup" in stats
        assert "backup_count" in stats
        assert "integrity_ok" in stats
        
        # Verify table counts
        assert stats["table_counts"]["user_contexts"] == 1
        assert stats["table_counts"]["user_interactions"] == 1
        
        # Verify integrity
        assert stats["integrity_ok"] is True
        
        # Verify backup info
        assert stats["backup_count"] == 1
        assert stats["last_backup"] is not None
    
    def test_vacuum_database(self, temp_db):
        """Test database vacuum operation"""
        db = PersonalDatabase(temp_db)
        
        # Add and delete data to create fragmentation
        for i in range(100):
            db.execute_update("""
                INSERT INTO user_contexts (user_id, context_data)
                VALUES (?, ?)
            """, (f"user_{i}", f'{{"data": "{i}"}}'))
        
        # Delete half the data
        db.execute_update("DELETE FROM user_contexts WHERE user_id LIKE 'user_5%'")
        
        # Vacuum should succeed
        success = db.vacuum_database()
        assert success is True
    
    def test_analyze_database(self, temp_db):
        """Test database analysis operation"""
        db = PersonalDatabase(temp_db)
        
        # Add some data
        db.execute_update("""
            INSERT INTO user_contexts (user_id, context_data)
            VALUES (?, ?)
        """, ("analyze_user", '{"test": "analyze_data"}'))
        
        # Analyze should succeed
        success = db.analyze_database()
        assert success is True
    
    def test_database_integrity_check(self, temp_db):
        """Test database integrity verification"""
        db = PersonalDatabase(temp_db)
        
        # Fresh database should pass integrity check
        integrity_ok = db._verify_database_integrity(temp_db)
        assert integrity_ok is True
    
    def test_database_metadata(self, temp_db):
        """Test database metadata storage and retrieval"""
        db = PersonalDatabase(temp_db)
        
        # Verify version metadata was set during initialization
        results = db.execute_query("""
            SELECT value FROM database_metadata WHERE key = 'version'
        """)
        
        assert len(results) == 1
        assert results[0]['value'] == str(db.CURRENT_VERSION)
    
    def test_concurrent_access(self, temp_db):
        """Test concurrent database access"""
        import threading
        import time
        
        db = PersonalDatabase(temp_db)
        results = []
        errors = []
        
        def worker(worker_id):
            try:
                for i in range(10):
                    db.execute_update("""
                        INSERT INTO user_contexts (user_id, context_data)
                        VALUES (?, ?)
                    """, (f"worker_{worker_id}_user_{i}", f'{{"worker": {worker_id}, "i": {i}}}'))
                    time.sleep(0.001)  # Small delay
                results.append(f"worker_{worker_id}_completed")
            except Exception as e:
                errors.append(f"worker_{worker_id}_error: {e}")
        
        # Start multiple worker threads
        threads = []
        for worker_id in range(3):
            thread = threading.Thread(target=worker, args=(worker_id,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify no errors occurred
        assert len(errors) == 0
        assert len(results) == 3
        
        # Verify all data was inserted
        count_results = db.execute_query("SELECT COUNT(*) as count FROM user_contexts")
        assert count_results[0]['count'] == 30  # 3 workers * 10 inserts each
    
    def test_database_migration_custom(self, temp_db):
        """Test custom database migration"""
        # Create a database with custom migrations
        class TestDatabase(PersonalDatabase):
            def _get_migrations(self):
                base_migrations = super()._get_migrations()
                base_migrations.append(
                    DatabaseMigration(
                        version=2,
                        description="Add test table",
                        up_sql="CREATE TABLE test_table (id INTEGER PRIMARY KEY, name TEXT);",
                        down_sql="DROP TABLE test_table;"
                    )
                )
                return base_migrations
        
        db = TestDatabase(temp_db)
        
        # Verify custom migration was applied
        with sqlite3.connect(temp_db) as conn:
            cursor = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='test_table'
            """)
            result = cursor.fetchone()
            assert result is not None
            
            # Verify migration was recorded
            cursor = conn.execute("""
                SELECT version, description FROM database_migrations 
                WHERE version = 2
            """)
            migration_record = cursor.fetchone()
            assert migration_record is not None
            assert migration_record[1] == "Add test table"
    
    def test_database_close(self, temp_db):
        """Test database close operation"""
        db = PersonalDatabase(temp_db)
        
        # Close should not raise any errors
        db.close()
        
        # Should still be able to create new connections after close
        with db.get_connection() as conn:
            cursor = conn.execute("SELECT 1")
            result = cursor.fetchone()
            assert result[0] == 1