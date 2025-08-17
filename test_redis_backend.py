#!/usr/bin/env python3
"""
Simple test script for Redis MCP Backend functionality
"""

import asyncio
import sys
import os
import json
from datetime import datetime

# Add app directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from redis_mcp_backend import RedisMCPBackend, RedisConfig, MessageQueueConfig
from mcp_models import MCPMessage, MCPMessageType, MCPMessagePriority


async def test_redis_connection():
    """Test Redis connection management"""
    print("Testing Redis connection management...")
    
    # Create backend with test configuration
    config = RedisConfig(
        url="redis://localhost:6379/1",  # Use test database
        max_connections=5,
        connection_timeout=5
    )
    
    backend = RedisMCPBackend(config=config)
    
    try:
        # Test connection
        print("  Attempting to connect to Redis...")
        success = await backend.connect()
        
        if success:
            print("  ✓ Successfully connected to Redis")
            
            # Test basic operations
            stats = await backend.get_queue_stats()
            print(f"  ✓ Retrieved queue stats: {len(stats)} keys")
            
            # Test disconnection
            await backend.disconnect()
            print("  ✓ Successfully disconnected from Redis")
            
            return True
        else:
            print("  ✗ Failed to connect to Redis")
            return False
            
    except Exception as e:
        print(f"  ✗ Error during Redis connection test: {e}")
        return False


async def test_message_operations():
    """Test message publishing and subscription"""
    print("\nTesting message operations...")
    
    config = RedisConfig(url="redis://localhost:6379/1")
    backend = RedisMCPBackend(config=config)
    
    try:
        # Connect
        if not await backend.connect():
            print("  ✗ Failed to connect to Redis")
            return False
        
        # Create test message
        message = MCPMessage(
            type=MCPMessageType.CONTEXT_UPDATE.value,
            source_agent="test_agent",
            target_agents=["target_agent"],
            payload={
                "context_data": {"test": "data"},
                "context_type": "test_context"
            },
            priority=MCPMessagePriority.NORMAL.value
        )
        
        print(f"  Created test message: {message.id}")
        
        # Test message publishing
        print("  Testing message publishing...")
        success = await backend.publish_message("test_topic", message)
        
        if success:
            print("  ✓ Successfully published message")
        else:
            print("  ✗ Failed to publish message")
            return False
        
        # Test message history retrieval
        print("  Testing message history retrieval...")
        history = await backend.get_message_history("test_topic", limit=10)
        print(f"  ✓ Retrieved {len(history)} messages from history")
        
        # Test subscription (basic setup)
        print("  Testing subscription setup...")
        
        received_messages = []
        
        async def test_callback(msg):
            received_messages.append(msg)
            print(f"    Received message: {msg.id}")
        
        subscription_id = await backend.subscribe_to_topic("test_topic", test_callback)
        print(f"  ✓ Created subscription: {subscription_id}")
        
        # Publish another message to test subscription
        message2 = MCPMessage(
            type=MCPMessageType.TASK_NOTIFICATION.value,
            source_agent="task_agent",
            target_agents=["worker_agent"],
            payload={
                "task_id": "task-123",
                "action": "start"
            }
        )
        
        await backend.publish_message("test_topic", message2)
        
        # Wait a bit for message processing
        await asyncio.sleep(1)
        
        # Unsubscribe
        await backend.unsubscribe_from_topic(subscription_id)
        print("  ✓ Successfully unsubscribed")
        
        # Cleanup
        await backend.disconnect()
        print("  ✓ Message operations test completed")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error during message operations test: {e}")
        await backend.disconnect()
        return False


async def test_cleanup_operations():
    """Test message cleanup and maintenance"""
    print("\nTesting cleanup operations...")
    
    config = RedisConfig(url="redis://localhost:6379/1")
    queue_config = MessageQueueConfig(
        default_ttl=60,
        cleanup_interval=10
    )
    
    backend = RedisMCPBackend(config=config, queue_config=queue_config)
    
    try:
        if not await backend.connect():
            print("  ✗ Failed to connect to Redis")
            return False
        
        # Create some test messages with short TTL
        for i in range(5):
            message = MCPMessage(
                type=MCPMessageType.HEARTBEAT.value,
                source_agent=f"agent_{i}",
                target_agents=["server"],
                payload={"agent_status": "active"},
                ttl=1  # Very short TTL for testing
            )
            
            await backend.publish_message(f"test_cleanup_{i}", message)
        
        print("  ✓ Created test messages with short TTL")
        
        # Wait for messages to expire
        await asyncio.sleep(2)
        
        # Run cleanup
        cleaned_count = await backend.cleanup_expired_messages()
        print(f"  ✓ Cleaned up {cleaned_count} expired messages")
        
        # Test statistics
        stats = await backend.get_queue_stats()
        print(f"  ✓ Retrieved statistics: {stats.get('backend_stats', {}).get('cleanup_operations', 0)} cleanup operations")
        
        await backend.disconnect()
        return True
        
    except Exception as e:
        print(f"  ✗ Error during cleanup operations test: {e}")
        await backend.disconnect()
        return False


async def test_error_handling():
    """Test error handling and recovery"""
    print("\nTesting error handling...")
    
    # Test with invalid Redis URL
    config = RedisConfig(url="redis://invalid-host:6379")
    backend = RedisMCPBackend(config=config)
    
    try:
        # This should fail gracefully
        success = await backend.connect()
        
        if not success:
            print("  ✓ Gracefully handled connection failure")
        else:
            print("  ✗ Unexpected connection success with invalid URL")
            await backend.disconnect()
            return False
        
        # Test operations on disconnected backend
        message = MCPMessage(
            type=MCPMessageType.HEARTBEAT.value,
            source_agent="test_agent",
            target_agents=["server"],
            payload={"status": "test"}
        )
        
        # This should fail gracefully
        success = await backend.publish_message("test_topic", message)
        
        if not success:
            print("  ✓ Gracefully handled publish failure when disconnected")
        else:
            print("  ✗ Unexpected publish success when disconnected")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error during error handling test: {e}")
        return False


async def main():
    """Run all tests"""
    print("Redis MCP Backend Test Suite")
    print("=" * 40)
    
    tests = [
        ("Redis Connection", test_redis_connection),
        ("Message Operations", test_message_operations),
        ("Cleanup Operations", test_cleanup_operations),
        ("Error Handling", test_error_handling)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  ✗ Test {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 40)
    print("Test Results:")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed!")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)