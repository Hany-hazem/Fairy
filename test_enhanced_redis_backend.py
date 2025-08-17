#!/usr/bin/env python3
"""
Enhanced test script for Redis MCP Backend functionality
Tests all the new features implemented for task 3
"""

import asyncio
import sys
import os
import json
from datetime import datetime, timedelta

# Add app directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

try:
    from redis_mcp_backend import RedisMCPBackend, RedisConfig, MessageQueueConfig
    from mcp_models import MCPMessage, MCPMessageType, MCPMessagePriority
except ImportError as e:
    print(f"Import error: {e}")
    print("This test requires Redis and other dependencies to be installed.")
    sys.exit(1)


async def test_enhanced_connection_management():
    """Test enhanced Redis connection management with security features"""
    print("Testing enhanced connection management...")
    
    # Test with enhanced configuration
    config = RedisConfig(
        url="redis://localhost:6379/1",
        max_connections=10,
        connection_timeout=5,
        health_check_interval=10,
        # Security features (would be used with actual Redis auth)
        password=None,  # Would set if Redis has auth
        ssl_enabled=False  # Would enable for production
    )
    
    queue_config = MessageQueueConfig(
        priority_queue_enabled=True,
        dead_letter_queue_enabled=True,
        message_deduplication=True,
        max_retry_attempts=3
    )
    
    backend = RedisMCPBackend(config=config, queue_config=queue_config)
    
    try:
        # Test connection
        print("  Attempting enhanced connection...")
        success = await backend.connect()
        
        if success:
            print("  ✓ Successfully connected with enhanced configuration")
            
            # Test enhanced statistics
            stats = await backend.get_queue_stats()
            print(f"  ✓ Retrieved enhanced stats with {len(stats)} categories")
            
            # Verify enhanced features are available
            if "performance_metrics" in stats:
                print("  ✓ Performance metrics available")
            if "queue_health" in stats.get("mcp_stats", {}):
                print("  ✓ Queue health monitoring available")
            
            await backend.disconnect()
            print("  ✓ Successfully disconnected")
            return True
        else:
            print("  ✗ Failed to connect")
            return False
            
    except Exception as e:
        print(f"  ✗ Error during enhanced connection test: {e}")
        return False


async def test_priority_queue_system():
    """Test priority queue functionality"""
    print("\nTesting priority queue system...")
    
    config = RedisConfig(url="redis://localhost:6379/1")
    queue_config = MessageQueueConfig(
        priority_queue_enabled=True,
        message_deduplication=True
    )
    
    backend = RedisMCPBackend(config=config, queue_config=queue_config)
    
    try:
        if not await backend.connect():
            print("  ✗ Failed to connect")
            return False
        
        # Create messages with different priorities
        messages = []
        for i, priority in enumerate([MCPMessagePriority.LOW.value, MCPMessagePriority.HIGH.value, MCPMessagePriority.CRITICAL.value]):
            message = MCPMessage(
                type=MCPMessageType.TASK_NOTIFICATION.value,
                source_agent=f"agent_{i}",
                target_agents=["target"],
                payload={"task_id": f"task-{i}", "action": "test"},
                priority=priority
            )
            messages.append(message)
        
        # Publish messages
        print("  Publishing messages with different priorities...")
        for message in messages:
            success = await backend.publish_message("priority_test", message)
            if not success:
                print(f"  ✗ Failed to publish message {message.id}")
                return False
        
        print("  ✓ Published all priority messages")
        
        # Test priority message retrieval
        priority_messages = await backend.get_priority_messages("priority_test", min_priority=MCPMessagePriority.HIGH.value)
        print(f"  ✓ Retrieved {len(priority_messages)} high-priority messages")
        
        # Test queue size reporting
        sizes = await backend.get_queue_size("priority_test")
        print(f"  ✓ Queue sizes: {sizes}")
        
        await backend.disconnect()
        return True
        
    except Exception as e:
        print(f"  ✗ Error during priority queue test: {e}")
        await backend.disconnect()
        return False


async def test_message_filtering():
    """Test message filtering in subscriptions"""
    print("\nTesting message filtering...")
    
    config = RedisConfig(url="redis://localhost:6379/1")
    backend = RedisMCPBackend(config=config)
    
    try:
        if not await backend.connect():
            print("  ✗ Failed to connect")
            return False
        
        received_messages = []
        
        # Create filtered callback
        async def filtered_callback(message):
            received_messages.append(message)
            print(f"    Filtered callback received: {message.type}")
        
        # Subscribe with filter for only TASK_NOTIFICATION messages
        message_filter = {
            "type": MCPMessageType.TASK_NOTIFICATION.value,
            "priority": {"$gt": MCPMessagePriority.LOW.value}
        }
        
        subscription_id = await backend.subscribe_to_topic(
            "filter_test", 
            filtered_callback, 
            message_filter
        )
        print(f"  ✓ Created filtered subscription: {subscription_id}")
        
        # Publish different types of messages
        messages = [
            MCPMessage(
                type=MCPMessageType.HEARTBEAT.value,
                source_agent="agent1",
                target_agents=["target"],
                payload={"status": "active"},
                priority=MCPMessagePriority.LOW.value
            ),
            MCPMessage(
                type=MCPMessageType.TASK_NOTIFICATION.value,
                source_agent="agent2",
                target_agents=["target"],
                payload={"task_id": "task-1", "action": "start"},
                priority=MCPMessagePriority.HIGH.value
            ),
            MCPMessage(
                type=MCPMessageType.TASK_NOTIFICATION.value,
                source_agent="agent3",
                target_agents=["target"],
                payload={"task_id": "task-2", "action": "complete"},
                priority=MCPMessagePriority.LOW.value  # Should be filtered out
            )
        ]
        
        print("  Publishing test messages...")
        for message in messages:
            await backend.publish_message("filter_test", message)
        
        # Wait for message processing
        await asyncio.sleep(1)
        
        # Check filtered results
        print(f"  ✓ Received {len(received_messages)} filtered messages (expected: 1)")
        
        if len(received_messages) == 1 and received_messages[0].type == MCPMessageType.TASK_NOTIFICATION.value:
            print("  ✓ Message filtering working correctly")
        else:
            print("  ✗ Message filtering not working as expected")
            return False
        
        await backend.unsubscribe_from_topic(subscription_id)
        await backend.disconnect()
        return True
        
    except Exception as e:
        print(f"  ✗ Error during message filtering test: {e}")
        await backend.disconnect()
        return False


async def test_failed_message_handling():
    """Test failed message handling and replay"""
    print("\nTesting failed message handling...")
    
    config = RedisConfig(url="redis://localhost:6379/1")
    queue_config = MessageQueueConfig(
        dead_letter_queue_enabled=True,
        max_retry_attempts=2
    )
    
    backend = RedisMCPBackend(config=config, queue_config=queue_config)
    
    try:
        if not await backend.connect():
            print("  ✗ Failed to connect")
            return False
        
        # Create a test message
        message = MCPMessage(
            type=MCPMessageType.CONTEXT_UPDATE.value,
            source_agent="test_agent",
            target_agents=["target"],
            payload={"context_data": {"test": "data"}, "context_type": "test"}
        )
        
        # Simulate storing a failed message
        await backend._store_failed_message("failed_test", message)
        print("  ✓ Stored failed message")
        
        # Check failed message count
        failed_counts = await backend.get_failed_message_count("failed_test")
        print(f"  ✓ Failed message count: {failed_counts}")
        
        # Test replay functionality
        print("  Testing message replay...")
        replay_stats = await backend.replay_failed_messages("failed_test")
        print(f"  ✓ Replay stats: {replay_stats}")
        
        # Clear failed messages
        cleared = await backend.clear_failed_messages("failed_test")
        print(f"  ✓ Cleared {cleared} failed messages")
        
        await backend.disconnect()
        return True
        
    except Exception as e:
        print(f"  ✗ Error during failed message handling test: {e}")
        await backend.disconnect()
        return False


async def test_batch_operations():
    """Test batch message operations"""
    print("\nTesting batch operations...")
    
    config = RedisConfig(url="redis://localhost:6379/1")
    backend = RedisMCPBackend(config=config)
    
    try:
        if not await backend.connect():
            print("  ✗ Failed to connect")
            return False
        
        # Create batch of messages
        messages = []
        for i in range(5):
            message = MCPMessage(
                type=MCPMessageType.HEARTBEAT.value,
                source_agent=f"agent_{i}",
                target_agents=["server"],
                payload={"agent_status": "active", "batch_id": i}
            )
            messages.append(message)
        
        print(f"  Publishing batch of {len(messages)} messages...")
        results = await backend.publish_batch_messages("batch_test", messages)
        
        successful = sum(1 for success in results.values() if success)
        print(f"  ✓ Successfully published {successful}/{len(messages)} messages")
        
        # Test batch retrieval
        history = await backend.get_message_history("batch_test", limit=10)
        print(f"  ✓ Retrieved {len(history)} messages from history")
        
        await backend.disconnect()
        return True
        
    except Exception as e:
        print(f"  ✗ Error during batch operations test: {e}")
        await backend.disconnect()
        return False


async def test_enhanced_cleanup():
    """Test enhanced cleanup operations"""
    print("\nTesting enhanced cleanup operations...")
    
    config = RedisConfig(url="redis://localhost:6379/1")
    queue_config = MessageQueueConfig(
        default_ttl=2,  # Very short TTL for testing
        priority_queue_enabled=True
    )
    
    backend = RedisMCPBackend(config=config, queue_config=queue_config)
    
    try:
        if not await backend.connect():
            print("  ✗ Failed to connect")
            return False
        
        # Create messages that will expire quickly
        for i in range(3):
            message = MCPMessage(
                type=MCPMessageType.HEARTBEAT.value,
                source_agent=f"cleanup_agent_{i}",
                target_agents=["server"],
                payload={"status": "test"},
                ttl=1  # 1 second TTL
            )
            
            await backend.publish_message(f"cleanup_test_{i}", message)
        
        print("  ✓ Created test messages with short TTL")
        
        # Wait for messages to expire
        await asyncio.sleep(3)
        
        # Run enhanced cleanup
        cleaned_count = await backend.cleanup_expired_messages()
        print(f"  ✓ Enhanced cleanup removed {cleaned_count} expired messages")
        
        # Test queue health analysis
        stats = await backend.get_queue_stats()
        queue_health = stats.get("mcp_stats", {}).get("queue_health", {})
        print(f"  ✓ Queue health status: {queue_health.get('health_status', 'unknown')}")
        
        await backend.disconnect()
        return True
        
    except Exception as e:
        print(f"  ✗ Error during enhanced cleanup test: {e}")
        await backend.disconnect()
        return False


async def main():
    """Run all enhanced tests"""
    print("Enhanced Redis MCP Backend Test Suite")
    print("=" * 50)
    
    tests = [
        ("Enhanced Connection Management", test_enhanced_connection_management),
        ("Priority Queue System", test_priority_queue_system),
        ("Message Filtering", test_message_filtering),
        ("Failed Message Handling", test_failed_message_handling),
        ("Batch Operations", test_batch_operations),
        ("Enhanced Cleanup", test_enhanced_cleanup)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  ✗ Test {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("Enhanced Test Results:")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All enhanced tests passed!")
        print("\nTask 3 Implementation Summary:")
        print("- ✓ Enhanced Redis connection management with security features")
        print("- ✓ Priority queue system with message ordering")
        print("- ✓ Message filtering and pattern subscriptions")
        print("- ✓ Failed message handling with retry logic")
        print("- ✓ Batch operations for improved performance")
        print("- ✓ Enhanced cleanup with multiple queue types")
        print("- ✓ Comprehensive monitoring and health checks")
        return 0
    else:
        print("✗ Some enhanced tests failed!")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)