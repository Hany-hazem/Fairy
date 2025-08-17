"""
Tests for Privacy and Security Manager
"""

import pytest
import pytest_asyncio
import tempfile
import os
import json
from datetime import datetime, timedelta

from app.privacy_security_manager import (
    PrivacySecurityManager, ConsentStatus, DataCategory
)
from app.personal_assistant_models import PermissionType


class TestPrivacySecurityManager:
    """Test cases for PrivacySecurityManager"""
    
    @pytest_asyncio.fixture
    async def privacy_manager(self):
        """Create a test privacy manager with temporary database"""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_file:
            db_path = tmp_file.name
        
        try:
            manager = PrivacySecurityManager(db_path)
            yield manager
        finally:
            # Cleanup
            if os.path.exists(db_path):
                os.unlink(db_path)
            # Also cleanup WAL files
            for ext in ["-wal", "-shm"]:
                wal_file = db_path + ext
                if os.path.exists(wal_file):
                    os.unlink(wal_file)
            # Cleanup encryption key file
            if os.path.exists("encryption.key"):
                os.unlink("encryption.key")
    
    @pytest.mark.asyncio
    async def test_request_permission_granted(self, privacy_manager):
        """Test requesting permission that gets granted"""
        user_id = "test_user"
        permission_type = PermissionType.FILE_READ
        
        granted = await privacy_manager.request_permission(
            user_id, permission_type, {"path": "/home/user/documents"}
        )
        
        # Should be granted for non-sensitive permissions
        assert granted is True
        
        # Verify permission is stored
        permission = await privacy_manager.get_permission(user_id, permission_type)
        assert permission is not None
        assert permission.granted is True
        assert permission.revoked is False
        assert permission.scope == {"path": "/home/user/documents"}
    
    @pytest.mark.asyncio
    async def test_request_permission_denied(self, privacy_manager):
        """Test requesting permission that gets denied"""
        user_id = "test_user"
        permission_type = PermissionType.SCREEN_MONITOR  # Sensitive permission
        
        granted = await privacy_manager.request_permission(
            user_id, permission_type
        )
        
        # Should be denied for sensitive permissions in test
        assert granted is False
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_permission(self, privacy_manager):
        """Test getting permission that doesn't exist"""
        user_id = "test_user"
        permission_type = PermissionType.AUTOMATION
        
        permission = await privacy_manager.get_permission(user_id, permission_type)
        
        assert permission is None
    
    @pytest.mark.asyncio
    async def test_revoke_permission(self, privacy_manager):
        """Test revoking a granted permission"""
        user_id = "test_user"
        permission_type = PermissionType.PERSONAL_DATA
        
        # Grant permission first
        await privacy_manager.request_permission(user_id, permission_type)
        
        # Verify it's granted
        permission = await privacy_manager.get_permission(user_id, permission_type)
        assert permission.granted is True
        
        # Revoke permission
        revoked = await privacy_manager.revoke_permission(user_id, permission_type)
        assert revoked is True
        
        # Verify it's revoked
        permission = await privacy_manager.get_permission(user_id, permission_type)
        assert permission.revoked is True
    
    @pytest.mark.asyncio
    async def test_check_permission_valid(self, privacy_manager):
        """Test checking a valid permission"""
        user_id = "test_user"
        permission_type = PermissionType.FILE_WRITE
        
        # Grant permission
        await privacy_manager.request_permission(user_id, permission_type)
        
        # Check permission
        has_permission = await privacy_manager.check_permission(user_id, permission_type)
        assert has_permission is True
    
    @pytest.mark.asyncio
    async def test_check_permission_invalid(self, privacy_manager):
        """Test checking an invalid permission"""
        user_id = "test_user"
        permission_type = PermissionType.LEARNING
        
        # Don't grant permission
        has_permission = await privacy_manager.check_permission(user_id, permission_type)
        assert has_permission is False
    
    @pytest.mark.asyncio
    async def test_permission_expiration(self, privacy_manager):
        """Test permission expiration"""
        user_id = "test_user"
        permission_type = PermissionType.FILE_READ
        
        # Grant permission with short expiration
        await privacy_manager.request_permission(
            user_id, permission_type, expires_in_days=0  # Expires immediately
        )
        
        # Manually set expiration to past
        with privacy_manager.get_connection() as conn:
            conn.execute("""
                UPDATE user_permissions 
                SET expires_at = ? 
                WHERE user_id = ? AND permission_type = ?
            """, (datetime.now() - timedelta(hours=1), user_id, permission_type.value))
            conn.commit()
        
        # Check permission should return False for expired permission
        has_permission = await privacy_manager.check_permission(user_id, permission_type)
        assert has_permission is False
    
    @pytest.mark.asyncio
    async def test_request_consent_granted(self, privacy_manager):
        """Test requesting consent that gets granted"""
        user_id = "test_user"
        data_category = DataCategory.INTERACTION_HISTORY
        
        status = await privacy_manager.request_consent(
            user_id, data_category, "To improve responses", retention_days=30
        )
        
        assert status == ConsentStatus.GRANTED
        
        # Verify consent is stored
        stored_status = await privacy_manager.get_consent_status(user_id, data_category)
        assert stored_status == ConsentStatus.GRANTED
    
    @pytest.mark.asyncio
    async def test_request_consent_denied(self, privacy_manager):
        """Test requesting consent that gets denied"""
        user_id = "test_user"
        data_category = DataCategory.SCREEN_CONTENT  # Sensitive category
        
        status = await privacy_manager.request_consent(
            user_id, data_category, "For context awareness"
        )
        
        assert status == ConsentStatus.DENIED
    
    @pytest.mark.asyncio
    async def test_get_consent_status_pending(self, privacy_manager):
        """Test getting consent status for non-existent consent"""
        user_id = "test_user"
        data_category = DataCategory.LEARNING_DATA
        
        status = await privacy_manager.get_consent_status(user_id, data_category)
        
        assert status == ConsentStatus.PENDING
    
    @pytest.mark.asyncio
    async def test_revoke_consent(self, privacy_manager):
        """Test revoking granted consent"""
        user_id = "test_user"
        data_category = DataCategory.PREFERENCES
        
        # Grant consent first
        await privacy_manager.request_consent(user_id, data_category, "For personalization")
        
        # Verify it's granted
        status = await privacy_manager.get_consent_status(user_id, data_category)
        assert status == ConsentStatus.GRANTED
        
        # Revoke consent
        revoked = await privacy_manager.revoke_consent(user_id, data_category)
        assert revoked is True
        
        # Verify it's revoked
        status = await privacy_manager.get_consent_status(user_id, data_category)
        assert status == ConsentStatus.REVOKED
    
    @pytest.mark.asyncio
    async def test_encrypt_decrypt_personal_data(self, privacy_manager):
        """Test encrypting and decrypting personal data"""
        user_id = "test_user"
        data_key = "user_profile"
        test_data = {
            "name": "John Doe",
            "email": "john@example.com",
            "preferences": {"theme": "dark", "language": "en"}
        }
        
        # Encrypt data
        encrypted = await privacy_manager.encrypt_personal_data(
            user_id, data_key, test_data, DataCategory.PERSONAL_INFO
        )
        assert encrypted is True
        
        # Decrypt data
        decrypted_data = await privacy_manager.decrypt_personal_data(user_id, data_key)
        
        assert decrypted_data is not None
        assert decrypted_data == test_data
        assert decrypted_data["name"] == "John Doe"
        assert decrypted_data["preferences"]["theme"] == "dark"
    
    @pytest.mark.asyncio
    async def test_decrypt_nonexistent_data(self, privacy_manager):
        """Test decrypting data that doesn't exist"""
        user_id = "test_user"
        data_key = "nonexistent_key"
        
        decrypted_data = await privacy_manager.decrypt_personal_data(user_id, data_key)
        
        assert decrypted_data is None
    
    @pytest.mark.asyncio
    async def test_request_data_deletion(self, privacy_manager):
        """Test requesting data deletion"""
        user_id = "test_user"
        categories = [DataCategory.INTERACTION_HISTORY, DataCategory.PERSONAL_INFO]
        
        # First, add some data to delete
        await privacy_manager.encrypt_personal_data(
            user_id, "test_data", {"test": "data"}, DataCategory.PERSONAL_INFO
        )
        
        # Request deletion
        request_id = await privacy_manager.request_data_deletion(
            user_id, categories, "Privacy concern"
        )
        
        assert request_id is not None
        assert len(request_id) > 0
        
        # Verify deletion request was processed (in test, it processes immediately)
        # The encrypted data should be deleted
        decrypted_data = await privacy_manager.decrypt_personal_data(user_id, "test_data")
        assert decrypted_data is None
    
    @pytest.mark.asyncio
    async def test_get_privacy_dashboard_data(self, privacy_manager):
        """Test getting privacy dashboard data"""
        user_id = "test_user"
        
        # Set up some permissions and consents
        await privacy_manager.request_permission(user_id, PermissionType.FILE_READ)
        await privacy_manager.request_permission(user_id, PermissionType.PERSONAL_DATA)
        await privacy_manager.request_consent(user_id, DataCategory.INTERACTION_HISTORY, "test")
        
        # Add some encrypted data
        await privacy_manager.encrypt_personal_data(
            user_id, "profile", {"name": "Test"}, DataCategory.PERSONAL_INFO
        )
        
        dashboard_data = await privacy_manager.get_privacy_dashboard_data(user_id)
        
        assert "permissions" in dashboard_data
        assert "consents" in dashboard_data
        assert "data_storage" in dashboard_data
        assert "last_updated" in dashboard_data
        
        # Check permissions
        permissions = dashboard_data["permissions"]
        assert permissions[PermissionType.FILE_READ.value]["granted"] is True
        assert permissions[PermissionType.PERSONAL_DATA.value]["granted"] is True
        
        # Check consents
        consents = dashboard_data["consents"]
        assert consents[DataCategory.INTERACTION_HISTORY.value] == ConsentStatus.GRANTED.value
        
        # Check data storage
        data_storage = dashboard_data["data_storage"]
        assert DataCategory.PERSONAL_INFO.value in data_storage
        assert data_storage[DataCategory.PERSONAL_INFO.value] == 1
    
    @pytest.mark.asyncio
    async def test_audit_logging(self, privacy_manager):
        """Test that audit events are logged"""
        user_id = "test_user"
        
        # Perform actions that should be logged
        await privacy_manager.request_permission(user_id, PermissionType.FILE_READ)
        await privacy_manager.revoke_permission(user_id, PermissionType.FILE_READ)
        await privacy_manager.request_consent(user_id, DataCategory.PREFERENCES, "test")
        await privacy_manager.encrypt_personal_data(
            user_id, "test", {"data": "value"}, DataCategory.PERSONAL_INFO
        )
        
        # Check audit log
        with privacy_manager.get_connection() as conn:
            cursor = conn.execute("""
                SELECT action, resource FROM audit_log 
                WHERE user_id = ? 
                ORDER BY timestamp
            """, (user_id,))
            
            audit_entries = cursor.fetchall()
            
            assert len(audit_entries) >= 4
            
            actions = [entry[0] for entry in audit_entries]
            assert "permission_requested" in actions
            assert "permission_granted" in actions
            assert "permission_revoked" in actions
            assert "consent_requested" in actions
            assert "data_encrypted" in actions
    
    @pytest.mark.asyncio
    async def test_multiple_users_isolation(self, privacy_manager):
        """Test that data is properly isolated between users"""
        user1_id = "user1"
        user2_id = "user2"
        
        # Grant different permissions to each user
        await privacy_manager.request_permission(user1_id, PermissionType.FILE_READ)
        await privacy_manager.request_permission(user2_id, PermissionType.FILE_WRITE)
        
        # Encrypt different data for each user
        await privacy_manager.encrypt_personal_data(
            user1_id, "profile", {"name": "User1"}, DataCategory.PERSONAL_INFO
        )
        await privacy_manager.encrypt_personal_data(
            user2_id, "profile", {"name": "User2"}, DataCategory.PERSONAL_INFO
        )
        
        # Verify user1 permissions
        has_read = await privacy_manager.check_permission(user1_id, PermissionType.FILE_READ)
        has_write = await privacy_manager.check_permission(user1_id, PermissionType.FILE_WRITE)
        assert has_read is True
        assert has_write is False
        
        # Verify user2 permissions
        has_read = await privacy_manager.check_permission(user2_id, PermissionType.FILE_READ)
        has_write = await privacy_manager.check_permission(user2_id, PermissionType.FILE_WRITE)
        assert has_read is False
        assert has_write is True
        
        # Verify data isolation
        user1_data = await privacy_manager.decrypt_personal_data(user1_id, "profile")
        user2_data = await privacy_manager.decrypt_personal_data(user2_id, "profile")
        
        assert user1_data["name"] == "User1"
        assert user2_data["name"] == "User2"
        
        # Verify user1 can't access user2's data
        cross_access = await privacy_manager.decrypt_personal_data(user1_id, "user2_profile")
        assert cross_access is None
    
    @pytest.mark.asyncio
    async def test_consent_expiration(self, privacy_manager):
        """Test consent expiration"""
        user_id = "test_user"
        data_category = DataCategory.LEARNING_DATA
        
        # Grant consent with short retention
        await privacy_manager.request_consent(
            user_id, data_category, "test", retention_days=0
        )
        
        # Manually set expiration to past
        with privacy_manager.get_connection() as conn:
            conn.execute("""
                UPDATE user_consent 
                SET expires_at = ? 
                WHERE user_id = ? AND data_category = ?
            """, (datetime.now() - timedelta(hours=1), user_id, data_category.value))
            conn.commit()
        
        # Check consent status should return EXPIRED
        status = await privacy_manager.get_consent_status(user_id, data_category)
        assert status == ConsentStatus.EXPIRED
    
    @pytest.mark.asyncio
    async def test_encryption_key_persistence(self, privacy_manager):
        """Test that encryption key is persistent across instances"""
        user_id = "test_user"
        data_key = "persistent_data"
        test_data = {"message": "This should persist"}
        
        # Encrypt data with first instance
        await privacy_manager.encrypt_personal_data(
            user_id, data_key, test_data, DataCategory.PERSONAL_INFO
        )
        
        # Create new instance (should use same key file)
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_file:
            db_path2 = tmp_file.name
        
        try:
            # Copy database to new location
            import shutil
            shutil.copy2(privacy_manager.db_path, db_path2)
            
            # Create new manager instance
            privacy_manager2 = PrivacySecurityManager(db_path2)
            
            # Should be able to decrypt with new instance
            decrypted_data = await privacy_manager2.decrypt_personal_data(user_id, data_key)
            
            assert decrypted_data is not None
            assert decrypted_data["message"] == "This should persist"
            
        finally:
            if os.path.exists(db_path2):
                os.unlink(db_path2)