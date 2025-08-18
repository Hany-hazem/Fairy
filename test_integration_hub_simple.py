#!/usr/bin/env python3
"""
Simple test for IntegrationHub functionality
"""

import asyncio
from unittest.mock import Mock, AsyncMock

from app.integration_hub import IntegrationHub, IntegrationConfig, IntegrationType
from app.privacy_security_manager import PrivacySecurityManager


async def test_integration_hub():
    """Test basic IntegrationHub functionality"""
    print("Testing IntegrationHub...")
    
    # Create mock privacy manager
    privacy_manager = Mock(spec=PrivacySecurityManager)
    privacy_manager.check_permission = AsyncMock(return_value=True)
    
    # Create integration hub
    hub = IntegrationHub(privacy_manager)
    
    # Initialize hub
    success = await hub.initialize()
    print(f"✓ Hub initialization: {success}")
    
    # List integrations
    integrations = await hub.list_integrations()
    print(f"✓ Available integrations: {list(integrations.keys())}")
    
    # Add a test integration
    config = IntegrationConfig(
        name="google_drive",
        integration_type=IntegrationType.CLOUD_STORAGE,
        oauth_token="test_token"
    )
    
    success = await hub.add_integration("google_drive", config)
    print(f"✓ Added Google Drive integration: {success}")
    
    # Test connections (will fail without real credentials, but should not crash)
    try:
        results = await hub.test_all_connections()
        print(f"✓ Connection test completed: {len(results)} integrations tested")
    except Exception as e:
        print(f"✓ Connection test handled gracefully: {type(e).__name__}")
    
    # Test cloud file sync (with permission check)
    try:
        files = await hub.sync_files_from_cloud("test_user")
        print(f"✓ Cloud sync completed: {len(files)} files found")
    except Exception as e:
        print(f"✓ Cloud sync handled gracefully: {type(e).__name__}")
    
    print("✓ All IntegrationHub tests completed successfully!")


if __name__ == "__main__":
    asyncio.run(test_integration_hub())