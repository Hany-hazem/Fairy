"""
Enhanced Privacy and Security Manager Tests

Tests for the enhanced privacy and security controls including granular permissions,
data encryption, consent management, data deletion, and privacy dashboard features.
"""

import pytest
import pytest_asyncio
import tempfile
import os
import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

from app.privacy_security_manager import (
    PrivacySecurityManager, ConsentStatus, DataCategory, PrivacyLevel, 
    DataRetentionPolicy
)
from app.personal_assistant_models import PermissionType


@pytest_asyncio.fixture
async def privacy_manager():
    """Create a privacy manager with temporary database"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as tmp_file:
        db_path = tmp_file.name
    
    manager = PrivacySecurityManager(db_path=db_path)
    yield manager
    
    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.mark.asyncio
class TestGranularPermissionControls:
    """Test granular permission controls"""
    
    async def test_request_permission_with_scope(self, privacy_manager):
        """Test requesting permission with specific scope"""
        user_id = "test_user"
        permission_type = PermissionType.FILE_READ
        scope = {"directories": ["/home/user/documents"], "file_types": [".txt", ".pdf"]}
        
        # Mock user approval
        with patch.object(privacy_manager, '_simulate_user_permission_response', return_value=True):
            granted = await privacy_manager.request_permission(
                user_id, permission_type, scope, expires_in_days=30
            )
        
        assert granted is True
        
        # Verify permission was stored with scope
        permission = await privacy_manager.get_permission(user_id, permission_type)
        assert permission is not None
        assert permission.granted is True
        assert permission.scope == scope
        assert permission.expires_at is not None
    
    async def test_permission_expiration(self, privacy_manager):
        """Test permission expiration handling"""
        user_id = "test_user"
        permission_type = PermissionType.SCREEN_MONITOR
        
        # Grant permission with 1-day expiration
        with patch.object(privacy_manager, '_simulate_user_permission_response', return_value=True):
            await privacy_manager.request_permission(
                user_id, permission_type, expires_in_days=1
            )
        
        # Verify permission is valid
        has_permission = await privacy_manager.check_permission(user_id, permission_type)
        assert has_permission is True
        
        # Mock expired permission by setting expires_at to past
        with privacy_manager.get_connection() as conn:
            conn.execute("""
                UPDATE user_permissions 
                SET expires_at = ?
                WHERE user_id = ? AND permission_type = ?
            """, (datetime.now() - timedelta(days=1), user_id, permission_type.value))
            conn.commit()
        
        # Check permission should now return False and expire it
        has_permission = await privacy_manager.check_permission(user_id, permission_type)
        assert has_permission is False
    
    async def test_revoke_permission(self, privacy_manager):
        """Test permission revocation"""
        user_id = "test_user"
        permission_type = PermissionType.FILE_READ
        
        # Grant permission
        with patch.object(privacy_manager, '_simulate_user_permission_response', return_value=True):
            await privacy_manager.request_permission(user_id, permission_type)
        
        # Verify permission is granted
        has_permission = await privacy_manager.check_permission(user_id, permission_type)
        assert has_permission is True
        
        # Revoke permission
        revoked = await privacy_manager.revoke_permission(user_id, permission_type)
        assert revoked is True
        
        # Verify permission is revoked
        has_permission = await privacy_manager.check_permission(user_id, permission_type)
        assert has_permission is False


@pytest.mark.asyncio
class TestDataEncryptionAndSecureStorage:
    """Test enhanced data encryption and secure storage"""
    
    async def test_encrypt_data_with_privacy_level(self, privacy_manager):
        """Test encrypting data with privacy level"""
        user_id = "test_user"
        data_key = "test_data"
        data = {"sensitive": "information", "user_id": user_id}
        data_category = DataCategory.PERSONAL_INFO
        privacy_level = PrivacyLevel.RESTRICTED
        
        # Grant consent first
        with patch.object(privacy_manager, '_simulate_user_consent_response', 
                         return_value=ConsentStatus.GRANTED):
            await privacy_manager.request_consent(user_id, data_category, "testing")
        
        # Encrypt data
        success = await privacy_manager.encrypt_personal_data(
            user_id, data_key, data, data_category, privacy_level
        )
        assert success is True
        
        # Verify data access was logged
        access_history = await privacy_manager.get_data_access_history(user_id)
        assert len(access_history) > 0
        assert access_history[0]["access_type"] == "write"
        assert access_history[0]["data_category"] == data_category.value
    
    async def test_decrypt_data_with_consent_check(self, privacy_manager):
        """Test decrypting data with consent verification"""
        user_id = "test_user"
        data_key = "test_data"
        data = {"test": "data"}
        data_category = DataCategory.INTERACTION_HISTORY
        
        # Grant consent and encrypt data
        with patch.object(privacy_manager, '_simulate_user_consent_response', 
                         return_value=ConsentStatus.GRANTED):
            await privacy_manager.request_consent(user_id, data_category, "testing")
        
        await privacy_manager.encrypt_personal_data(user_id, data_key, data, data_category)
        
        # Decrypt data
        decrypted_data = await privacy_manager.decrypt_personal_data(
            user_id, data_key, purpose="testing"
        )
        assert decrypted_data == data
        
        # Verify read access was logged
        access_history = await privacy_manager.get_data_access_history(user_id)
        read_accesses = [a for a in access_history if a["access_type"] == "read"]
        assert len(read_accesses) > 0
        assert read_accesses[0]["purpose"] == "testing"
    
    async def test_decrypt_without_consent_fails(self, privacy_manager):
        """Test that decryption fails without valid consent"""
        user_id = "test_user"
        data_key = "test_data"
        data = {"test": "data"}
        data_category = DataCategory.SCREEN_CONTENT
        
        # Grant consent, encrypt data, then revoke consent
        with patch.object(privacy_manager, '_simulate_user_consent_response', 
                         return_value=ConsentStatus.GRANTED):
            await privacy_manager.request_consent(user_id, data_category, "testing")
        
        await privacy_manager.encrypt_personal_data(user_id, data_key, data, data_category)
        await privacy_manager.revoke_consent(user_id, data_category)
        
        # Attempt to decrypt should fail
        decrypted_data = await privacy_manager.decrypt_personal_data(user_id, data_key)
        assert decrypted_data is None
        
        # Verify privacy violation was reported
        violations = await privacy_manager.get_privacy_violations(user_id, resolved=False)
        assert len(violations) > 0
        assert violations[0]["violation_type"] == "unauthorized_data_access"


@pytest.mark.asyncio
class TestConsentManagement:
    """Test comprehensive consent management"""
    
    async def test_consent_with_retention_period(self, privacy_manager):
        """Test consent with retention period"""
        user_id = "test_user"
        data_category = DataCategory.LEARNING_DATA
        
        with patch.object(privacy_manager, '_simulate_user_consent_response', 
                         return_value=ConsentStatus.GRANTED):
            status = await privacy_manager.request_consent(
                user_id, data_category, "machine learning", retention_days=30
            )
        
        assert status == ConsentStatus.GRANTED
        
        # Verify consent was recorded with expiration
        consent_status = await privacy_manager.get_consent_status(user_id, data_category)
        assert consent_status == ConsentStatus.GRANTED
    
    async def test_consent_revocation(self, privacy_manager):
        """Test consent revocation"""
        user_id = "test_user"
        data_category = DataCategory.FILE_ACCESS
        
        # Grant consent
        with patch.object(privacy_manager, '_simulate_user_consent_response', 
                         return_value=ConsentStatus.GRANTED):
            await privacy_manager.request_consent(user_id, data_category, "file management")
        
        # Revoke consent
        revoked = await privacy_manager.revoke_consent(user_id, data_category)
        assert revoked is True
        
        # Verify consent status
        status = await privacy_manager.get_consent_status(user_id, data_category)
        assert status == ConsentStatus.REVOKED


@pytest.mark.asyncio
class TestDataDeletion:
    """Test comprehensive data deletion"""
    
    async def test_complete_data_deletion(self, privacy_manager):
        """Test complete data deletion by category"""
        user_id = "test_user"
        data_categories = [DataCategory.PERSONAL_INFO, DataCategory.INTERACTION_HISTORY]
        
        # Create some test data
        for category in data_categories:
            with patch.object(privacy_manager, '_simulate_user_consent_response', 
                             return_value=ConsentStatus.GRANTED):
                await privacy_manager.request_consent(user_id, category, "testing")
            
            await privacy_manager.encrypt_personal_data(
                user_id, f"test_data_{category.value}", {"test": "data"}, category
            )
        
        # Request data deletion
        request_id = await privacy_manager.request_data_deletion(
            user_id, data_categories, "user request"
        )
        assert request_id is not None
        
        # Verify data was deleted
        for category in data_categories:
            decrypted_data = await privacy_manager.decrypt_personal_data(
                user_id, f"test_data_{category.value}"
            )
            assert decrypted_data is None
    
    async def test_data_deletion_with_audit_trail(self, privacy_manager):
        """Test that data deletion maintains audit trail"""
        user_id = "test_user"
        data_category = DataCategory.PREFERENCES
        
        # Create and delete data
        with patch.object(privacy_manager, '_simulate_user_consent_response', 
                         return_value=ConsentStatus.GRANTED):
            await privacy_manager.request_consent(user_id, data_category, "testing")
        
        await privacy_manager.encrypt_personal_data(
            user_id, "test_prefs", {"theme": "dark"}, data_category
        )
        
        await privacy_manager.request_data_deletion(user_id, [data_category])
        
        # Verify audit trail exists
        with privacy_manager.get_connection() as conn:
            cursor = conn.execute("""
                SELECT COUNT(*) FROM audit_log 
                WHERE user_id = ? AND action LIKE '%deletion%'
            """, (user_id,))
            deletion_events = cursor.fetchone()[0]
            assert deletion_events > 0


@pytest.mark.asyncio
class TestPrivacyDashboard:
    """Test privacy dashboard and transparency features"""
    
    async def test_enhanced_privacy_dashboard(self, privacy_manager):
        """Test enhanced privacy dashboard data"""
        user_id = "test_user"
        
        # Set up some test data
        await privacy_manager.set_privacy_setting(
            user_id, "data_sharing", False, PrivacyLevel.RESTRICTED
        )
        
        await privacy_manager.set_data_retention_policy(
            user_id, DataCategory.INTERACTION_HISTORY, 
            DataRetentionPolicy.SHORT_TERM, auto_delete=True
        )
        
        # Get dashboard data
        dashboard_data = await privacy_manager.get_enhanced_privacy_dashboard_data(user_id)
        
        assert "privacy_settings" in dashboard_data
        assert "retention_policies" in dashboard_data
        assert "privacy_score" in dashboard_data
        assert "transparency_report" in dashboard_data
        
        # Verify privacy settings
        assert "data_sharing" in dashboard_data["privacy_settings"]
        assert dashboard_data["privacy_settings"]["data_sharing"]["value"] is False
        
        # Verify retention policies
        assert DataCategory.INTERACTION_HISTORY.value in dashboard_data["retention_policies"]
        policy = dashboard_data["retention_policies"][DataCategory.INTERACTION_HISTORY.value]
        assert policy["policy"] == DataRetentionPolicy.SHORT_TERM
        assert policy["auto_delete"] is True
    
    async def test_privacy_score_calculation(self, privacy_manager):
        """Test privacy score calculation"""
        user_id = "test_user"
        
        # Set up a user with good privacy practices
        await privacy_manager.set_data_retention_policy(
            user_id, DataCategory.SCREEN_CONTENT, DataRetentionPolicy.SESSION_ONLY
        )
        await privacy_manager.set_data_retention_policy(
            user_id, DataCategory.LOCATION_DATA, DataRetentionPolicy.SHORT_TERM
        )
        
        # Create some encrypted data
        with patch.object(privacy_manager, '_simulate_user_consent_response', 
                         return_value=ConsentStatus.GRANTED):
            await privacy_manager.request_consent(user_id, DataCategory.PERSONAL_INFO, "testing")
        
        await privacy_manager.encrypt_personal_data(
            user_id, "test_data", {"test": "data"}, DataCategory.PERSONAL_INFO
        )
        
        dashboard_data = await privacy_manager.get_enhanced_privacy_dashboard_data(user_id)
        privacy_score = dashboard_data["privacy_score"]
        
        assert "score" in privacy_score
        assert "factors" in privacy_score
        assert "recommendations" in privacy_score
        assert isinstance(privacy_score["score"], int)
        assert 0 <= privacy_score["score"] <= 100
    
    async def test_data_access_transparency(self, privacy_manager):
        """Test data access transparency features"""
        user_id = "test_user"
        data_category = DataCategory.FILE_ACCESS
        
        # Log some data accesses
        await privacy_manager.log_data_access(
            user_id, data_category, "read", "document.txt", "user_request"
        )
        await privacy_manager.log_data_access(
            user_id, data_category, "write", "document.txt", "auto_save"
        )
        
        # Get access history
        access_history = await privacy_manager.get_data_access_history(user_id, data_category)
        
        assert len(access_history) == 2
        assert access_history[0]["access_type"] in ["read", "write"]
        assert access_history[0]["data_key"] == "document.txt"
        assert access_history[0]["purpose"] in ["user_request", "auto_save"]


@pytest.mark.asyncio
class TestPrivacyViolations:
    """Test privacy violation reporting and resolution"""
    
    async def test_report_privacy_violation(self, privacy_manager):
        """Test reporting privacy violations"""
        user_id = "test_user"
        
        violation_id = await privacy_manager.report_privacy_violation(
            user_id, "unauthorized_access", 
            "System accessed data without permission", "high"
        )
        
        assert violation_id is not None
        
        # Verify violation was recorded
        violations = await privacy_manager.get_privacy_violations(user_id, resolved=False)
        assert len(violations) == 1
        assert violations[0]["violation_type"] == "unauthorized_access"
        assert violations[0]["severity"] == "high"
        assert violations[0]["resolved"] is False
    
    async def test_resolve_privacy_violation(self, privacy_manager):
        """Test resolving privacy violations"""
        user_id = "test_user"
        
        # Report violation
        violation_id = await privacy_manager.report_privacy_violation(
            user_id, "data_leak", "Data was exposed", "medium"
        )
        
        # Resolve violation
        resolved = await privacy_manager.resolve_privacy_violation(
            violation_id, "Fixed access controls and notified user"
        )
        assert resolved is True
        
        # Verify resolution
        violations = await privacy_manager.get_privacy_violations(user_id, resolved=True)
        assert len(violations) == 1
        assert violations[0]["resolved"] is True
        assert "Fixed access controls" in violations[0]["resolution_details"]


@pytest.mark.asyncio
class TestImmediatePrivacySettingsApplication:
    """Test immediate application of privacy settings changes"""
    
    async def test_privacy_settings_immediate_application(self, privacy_manager):
        """Test that privacy settings are applied immediately"""
        user_id = "test_user"
        
        # Add a mock listener
        listener_called = False
        listener_args = None
        
        async def mock_listener(user_id, setting_key, old_value, new_value):
            nonlocal listener_called, listener_args
            listener_called = True
            listener_args = (user_id, setting_key, old_value, new_value)
        
        privacy_manager.add_privacy_change_listener(mock_listener)
        
        # Set privacy setting
        success = await privacy_manager.set_privacy_setting(
            user_id, "screen_monitoring", False, PrivacyLevel.RESTRICTED
        )
        assert success is True
        
        # Verify listener was called
        assert listener_called is True
        assert listener_args[0] == user_id
        assert listener_args[1] == "screen_monitoring"
        assert listener_args[3] is False
    
    async def test_bulk_privacy_settings_application(self, privacy_manager):
        """Test applying multiple privacy settings immediately"""
        user_id = "test_user"
        
        settings = {
            "data_sharing": False,
            "analytics": False,
            "personalization": True,
            "retention_period": 30
        }
        
        success = await privacy_manager.apply_privacy_settings_immediately(user_id, settings)
        assert success is True
        
        # Verify all settings were applied
        for key, expected_value in settings.items():
            actual_value = await privacy_manager.get_privacy_setting(user_id, key)
            assert actual_value == expected_value


@pytest.mark.asyncio
class TestDataRetentionPolicies:
    """Test data retention policy management"""
    
    async def test_set_custom_retention_policy(self, privacy_manager):
        """Test setting custom retention policies"""
        user_id = "test_user"
        data_category = DataCategory.LEARNING_DATA
        
        success = await privacy_manager.set_data_retention_policy(
            user_id, data_category, DataRetentionPolicy.MEDIUM_TERM,
            custom_days=90, auto_delete=True
        )
        assert success is True
        
        # Verify policy was set
        policy = await privacy_manager.get_data_retention_policy(user_id, data_category)
        assert policy["policy"] == DataRetentionPolicy.MEDIUM_TERM
        assert policy["custom_days"] == 90
        assert policy["auto_delete"] is True
    
    async def test_default_retention_policies(self, privacy_manager):
        """Test default retention policies"""
        user_id = "test_user"
        
        # Get policy for category without custom setting
        policy = await privacy_manager.get_data_retention_policy(
            user_id, DataCategory.SCREEN_CONTENT
        )
        
        # Should return default policy
        assert policy["policy"] == DataRetentionPolicy.SESSION_ONLY
        assert policy["custom_days"] is None
        assert policy["auto_delete"] is False
        assert policy["updated_at"] is None


if __name__ == "__main__":
    pytest.main([__file__])