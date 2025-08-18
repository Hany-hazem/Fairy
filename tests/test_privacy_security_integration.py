"""
Privacy and Security Integration Tests

Integration tests for the enhanced privacy and security controls,
demonstrating how all components work together.
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
class TestPrivacySecurityIntegration:
    """Integration tests for privacy and security features"""
    
    async def test_complete_privacy_workflow(self, privacy_manager):
        """Test complete privacy workflow from setup to data deletion"""
        user_id = "integration_test_user"
        
        # Step 1: Set up privacy preferences
        privacy_settings = {
            "data_sharing": False,
            "analytics_enabled": False,
            "personalization": True,
            "screen_monitoring": False
        }
        
        for key, value in privacy_settings.items():
            success = await privacy_manager.set_privacy_setting(
                user_id, key, value, PrivacyLevel.RESTRICTED
            )
            assert success, f"Failed to set privacy setting {key}"
        
        # Step 2: Configure data retention policies
        retention_configs = [
            (DataCategory.SCREEN_CONTENT, DataRetentionPolicy.SESSION_ONLY, True),
            (DataCategory.INTERACTION_HISTORY, DataRetentionPolicy.SHORT_TERM, False),
            (DataCategory.PERSONAL_INFO, DataRetentionPolicy.PERMANENT, False),
        ]
        
        for category, policy, auto_delete in retention_configs:
            success = await privacy_manager.set_data_retention_policy(
                user_id, category, policy, auto_delete=auto_delete
            )
            assert success, f"Failed to set retention policy for {category}"
        
        # Step 3: Grant consent for data processing
        with patch.object(privacy_manager, '_simulate_user_consent_response', 
                         return_value=ConsentStatus.GRANTED):
            for category in [DataCategory.PERSONAL_INFO, DataCategory.INTERACTION_HISTORY]:
                status = await privacy_manager.request_consent(
                    user_id, category, f"Testing {category.value}", retention_days=30
                )
                assert status == ConsentStatus.GRANTED, f"Consent not granted for {category}"
        
        # Step 4: Store encrypted data
        test_data = [
            ("personal_info", {"name": "Test User", "email": "test@example.com"}, DataCategory.PERSONAL_INFO),
            ("interaction_1", {"query": "Hello", "response": "Hi there!"}, DataCategory.INTERACTION_HISTORY),
            ("interaction_2", {"query": "Help me", "response": "How can I help?"}, DataCategory.INTERACTION_HISTORY),
        ]
        
        for data_key, data, category in test_data:
            success = await privacy_manager.encrypt_personal_data(
                user_id, data_key, data, category, PrivacyLevel.CONFIDENTIAL
            )
            assert success, f"Failed to encrypt data {data_key}"
        
        # Step 5: Verify data access logging
        access_history = await privacy_manager.get_data_access_history(user_id)
        assert len(access_history) >= len(test_data), "Not all data accesses were logged"
        
        # Step 6: Get comprehensive privacy dashboard
        dashboard = await privacy_manager.get_enhanced_privacy_dashboard_data(user_id)
        
        # Verify dashboard contains all expected sections
        expected_sections = [
            "permissions", "consents", "data_storage", "privacy_settings",
            "retention_policies", "recent_access_history", "privacy_score",
            "transparency_report"
        ]
        
        for section in expected_sections:
            assert section in dashboard, f"Dashboard missing section: {section}"
        
        # Verify privacy settings in dashboard
        for key, expected_value in privacy_settings.items():
            assert key in dashboard["privacy_settings"], f"Privacy setting {key} not in dashboard"
            assert dashboard["privacy_settings"][key]["value"] == expected_value, \
                f"Privacy setting {key} has wrong value"
        
        # Verify retention policies in dashboard
        for category, expected_policy, expected_auto_delete in retention_configs:
            policy_data = dashboard["retention_policies"][category.value]
            assert policy_data["policy"] == expected_policy, \
                f"Retention policy for {category} is incorrect"
            assert policy_data["auto_delete"] == expected_auto_delete, \
                f"Auto delete setting for {category} is incorrect"
        
        # Step 7: Test privacy score calculation
        privacy_score = dashboard["privacy_score"]
        assert "score" in privacy_score, "Privacy score missing"
        assert "factors" in privacy_score, "Privacy score factors missing"
        assert "recommendations" in privacy_score, "Privacy score recommendations missing"
        assert isinstance(privacy_score["score"], int), "Privacy score should be integer"
        assert 0 <= privacy_score["score"] <= 100, "Privacy score should be 0-100"
        
        # Step 8: Test data retrieval with consent verification
        for data_key, expected_data, category in test_data:
            retrieved_data = await privacy_manager.decrypt_personal_data(
                user_id, data_key, purpose="integration_test"
            )
            assert retrieved_data == expected_data, f"Retrieved data for {data_key} doesn't match"
        
        # Step 9: Revoke consent and verify access is blocked
        revoked = await privacy_manager.revoke_consent(user_id, DataCategory.INTERACTION_HISTORY)
        assert revoked, "Failed to revoke consent"
        
        # Try to access revoked data - should fail
        blocked_data = await privacy_manager.decrypt_personal_data(
            user_id, "interaction_1", purpose="should_fail"
        )
        assert blocked_data is None, "Data access should be blocked after consent revocation"
        
        # Verify privacy violation was reported
        violations = await privacy_manager.get_privacy_violations(user_id, resolved=False)
        assert len(violations) > 0, "Privacy violation should be reported for blocked access"
        
        # Step 10: Test data deletion
        deletion_categories = [DataCategory.INTERACTION_HISTORY, DataCategory.PERSONAL_INFO]
        request_id = await privacy_manager.request_data_deletion(
            user_id, deletion_categories, "Integration test cleanup"
        )
        assert request_id is not None, "Data deletion request failed"
        
        # Verify data was deleted
        for data_key, _, category in test_data:
            if category in deletion_categories:
                deleted_data = await privacy_manager.decrypt_personal_data(user_id, data_key)
                assert deleted_data is None, f"Data {data_key} should be deleted"
        
        # Step 11: Verify audit trail exists
        with privacy_manager.get_connection() as conn:
            cursor = conn.execute("""
                SELECT COUNT(*) FROM audit_log 
                WHERE user_id = ?
            """, (user_id,))
            audit_count = cursor.fetchone()[0]
            assert audit_count > 0, "Audit trail should contain events"
    
    async def test_privacy_settings_immediate_application(self, privacy_manager):
        """Test that privacy settings are applied immediately across the system"""
        user_id = "immediate_test_user"
        
        # Set up listener to track changes
        changes_applied = []
        
        async def change_listener(user_id, setting_key, old_value, new_value):
            changes_applied.append({
                "user_id": user_id,
                "setting_key": setting_key,
                "old_value": old_value,
                "new_value": new_value,
                "timestamp": datetime.now()
            })
        
        privacy_manager.add_privacy_change_listener(change_listener)
        
        # Apply multiple settings
        settings_to_apply = {
            "screen_monitoring": False,
            "data_retention_days": 30,
            "auto_delete_enabled": True,
            "analytics_opt_out": True
        }
        
        success = await privacy_manager.apply_privacy_settings_immediately(
            user_id, settings_to_apply
        )
        assert success, "Failed to apply privacy settings immediately"
        
        # Verify all changes were tracked
        assert len(changes_applied) == len(settings_to_apply), \
            "Not all privacy setting changes were tracked"
        
        # Verify settings were actually applied
        for setting_key, expected_value in settings_to_apply.items():
            actual_value = await privacy_manager.get_privacy_setting(user_id, setting_key)
            assert actual_value == expected_value, \
                f"Setting {setting_key} was not applied correctly"
        
        # Verify change notifications contain correct data
        for change in changes_applied:
            assert change["user_id"] == user_id, "Change notification has wrong user ID"
            assert change["setting_key"] in settings_to_apply, \
                "Change notification for unexpected setting"
            assert change["new_value"] == settings_to_apply[change["setting_key"]], \
                "Change notification has wrong new value"
    
    async def test_comprehensive_data_protection(self, privacy_manager):
        """Test comprehensive data protection features"""
        user_id = "protection_test_user"
        
        # Test different privacy levels
        test_cases = [
            ("public_data", {"info": "public information"}, PrivacyLevel.PUBLIC),
            ("internal_data", {"info": "internal information"}, PrivacyLevel.INTERNAL),
            ("confidential_data", {"info": "confidential information"}, PrivacyLevel.CONFIDENTIAL),
            ("restricted_data", {"info": "restricted information"}, PrivacyLevel.RESTRICTED),
        ]
        
        # Grant consent for all data categories
        with patch.object(privacy_manager, '_simulate_user_consent_response', 
                         return_value=ConsentStatus.GRANTED):
            for category in DataCategory:
                await privacy_manager.request_consent(user_id, category, "comprehensive test")
        
        # Store data with different privacy levels
        for data_key, data, privacy_level in test_cases:
            success = await privacy_manager.encrypt_personal_data(
                user_id, data_key, data, DataCategory.PERSONAL_INFO, privacy_level
            )
            assert success, f"Failed to encrypt {privacy_level.value} data"
        
        # Verify all data can be retrieved
        for data_key, expected_data, privacy_level in test_cases:
            retrieved_data = await privacy_manager.decrypt_personal_data(
                user_id, data_key, purpose=f"test_{privacy_level.value}"
            )
            assert retrieved_data == expected_data, \
                f"Failed to retrieve {privacy_level.value} data"
        
        # Test data access transparency
        access_history = await privacy_manager.get_data_access_history(user_id)
        
        # Should have write and read access for each test case
        expected_accesses = len(test_cases) * 2  # write + read for each
        assert len(access_history) >= expected_accesses, \
            f"Expected at least {expected_accesses} access logs, got {len(access_history)}"
        
        # Verify access logs contain purpose information
        read_accesses = [a for a in access_history if a["access_type"] == "read"]
        for access in read_accesses:
            assert access["purpose"] is not None, "Access log missing purpose"
            assert access["purpose"].startswith("test_"), "Access log has unexpected purpose"
    
    async def test_privacy_violation_workflow(self, privacy_manager):
        """Test complete privacy violation reporting and resolution workflow"""
        user_id = "violation_test_user"
        
        # Report multiple violations of different severities
        violations_to_report = [
            ("unauthorized_access", "System accessed data without permission", "high"),
            ("data_leak", "Data was exposed in logs", "medium"),
            ("consent_violation", "Data processed without consent", "high"),
            ("retention_violation", "Data kept beyond retention period", "low"),
        ]
        
        violation_ids = []
        for violation_type, description, severity in violations_to_report:
            violation_id = await privacy_manager.report_privacy_violation(
                user_id, violation_type, description, severity
            )
            assert violation_id is not None, f"Failed to report {violation_type}"
            violation_ids.append(violation_id)
        
        # Verify all violations were recorded
        all_violations = await privacy_manager.get_privacy_violations(user_id)
        assert len(all_violations) == len(violations_to_report), \
            "Not all violations were recorded"
        
        # Verify unresolved violations
        unresolved = await privacy_manager.get_privacy_violations(user_id, resolved=False)
        assert len(unresolved) == len(violations_to_report), \
            "All violations should be unresolved initially"
        
        # Resolve some violations
        resolutions = [
            (violation_ids[0], "Fixed access controls and updated permissions"),
            (violation_ids[2], "Updated consent management system"),
        ]
        
        for violation_id, resolution in resolutions:
            resolved = await privacy_manager.resolve_privacy_violation(
                violation_id, resolution
            )
            assert resolved, f"Failed to resolve violation {violation_id}"
        
        # Verify resolution status
        resolved_violations = await privacy_manager.get_privacy_violations(user_id, resolved=True)
        assert len(resolved_violations) == len(resolutions), \
            "Wrong number of resolved violations"
        
        unresolved_violations = await privacy_manager.get_privacy_violations(user_id, resolved=False)
        expected_unresolved = len(violations_to_report) - len(resolutions)
        assert len(unresolved_violations) == expected_unresolved, \
            "Wrong number of unresolved violations"
        
        # Verify resolution details
        for resolved_violation in resolved_violations:
            assert resolved_violation["resolved"] is True, "Violation should be marked as resolved"
            assert resolved_violation["resolution_details"] is not None, \
                "Resolved violation should have resolution details"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])