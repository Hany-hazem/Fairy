#!/usr/bin/env python3
"""
MCP Server Infrastructure Demo

This script demonstrates the enhanced MCP server functionality including:
- Server initialization and agent registration
- Message handling and routing
- Redis backend integration
- Error handling and recovery
"""

import asyncio
import logging
import sys
import uuid
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.mcp_integration import MCPIntegration
from app.mcp_server import MCPMessage, MCPMessageType, MCPMessagePriority
from app.mcp_message_handler import MessageRoutingRule
from app.mcp_error_handler import ErrorContext, ErrorCategory

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DemoAgent:
    """Demo agent for testing MCP functionality"""
    
    def __init__(self, agent_id: str, mcp_integration: MCPIntegration):
        self.agent_id = agent_id
        self.mcp = mcp_integration
        self.connection_id = None
        self.message_count = 0
    
    async def register(self):
        """Register agent with MCP server"""
        capabilities = [
            {
                "name": "message_processing",
                "description": "Process various message types",
                "message_types": [
                    MCPMessageType.CONTEXT_UPDATE.value,
                    MCPMessageType.TASK_NOTIFICATION.value,
                    MCPMessageType.AGENT_REQUEST.value
                ],
                "parameters": {
                    "max_concurrent_messages": 10,
                    "supported_formats": ["json", "text"]
                }
            }
        ]
        
        self.connection_id = await self.mcp.register_agent(self.agent_id, capabilities)
        if self.connection_id:
            logger.info(f"Agent {self.agent_id} registered with connection ID: {self.connection_id}")
            return True
        else:
            logger.error(f"Failed to register agent {self.agent_id}")
            return False
    
    async def unregister(self):
        """Unregister agent from MCP server"""
        if self.connection_id:
            success = await self.mcp.unregister_agent(self.agent_id)
            if success:
                logger.info(f"Agent {self.agent_id} unregistered successfully")
            return success
        return True
    
    async def send_message(self, target_agent: str, message_type: str, payload: dict):
        """Send a message to another agent"""
        message = MCPMessage(
            id=str(uuid.uuid4()),
            type=message_type,
            source_agent=self.agent_id,
            target_agents=[target_agent],
            payload=payload,
            timestamp=datetime.utcnow(),
            priority=MCPMessagePriority.NORMAL.value
        )
        
        success = await self.mcp.send_message(message)
        if success:
            logger.info(f"Agent {self.agent_id} sent message {message.id} to {target_agent}")
        else:
            logger.error(f"Agent {self.agent_id} failed to send message to {target_agent}")
        
        return success
    
    async def broadcast_message(self, message_type: str, payload: dict):
        """Broadcast a message to all agents"""
        message = MCPMessage(
            id=str(uuid.uuid4()),
            type=message_type,
            source_agent=self.agent_id,
            target_agents=[],  # Will be populated by broadcast
            payload=payload,
            timestamp=datetime.utcnow(),
            priority=MCPMessagePriority.NORMAL.value
        )
        
        count = await self.mcp.broadcast_message(message)
        logger.info(f"Agent {self.agent_id} broadcast message {message.id} to {count} agents")
        return count
    
    async def message_handler(self, message: MCPMessage):
        """Handle received messages"""
        self.message_count += 1
        logger.info(f"Agent {self.agent_id} received message {message.id} from {message.source_agent}")
        logger.info(f"Message type: {message.type}, Payload: {message.payload}")
        
        # Send acknowledgment for certain message types
        if message.requires_ack:
            ack_message = MCPMessage(
                id=str(uuid.uuid4()),
                type=MCPMessageType.AGENT_RESPONSE.value,
                source_agent=self.agent_id,
                target_agents=[message.source_agent],
                payload={"status": "acknowledged", "original_message_id": message.id},
                timestamp=datetime.utcnow(),
                correlation_id=message.id
            )
            
            await self.mcp.send_message(ack_message)


async def demo_basic_functionality():
    """Demonstrate basic MCP server functionality"""
    logger.info("=== Demo: Basic MCP Server Functionality ===")
    
    # Initialize MCP integration
    mcp = MCPIntegration()
    
    try:
        # Initialize the system
        logger.info("Initializing MCP system...")
        success = await mcp.initialize()
        if not success:
            logger.error("Failed to initialize MCP system")
            return
        
        # Create demo agents
        agent1 = DemoAgent("demo_agent_1", mcp)
        agent2 = DemoAgent("demo_agent_2", mcp)
        
        # Register agents
        logger.info("Registering demo agents...")
        await agent1.register()
        await agent2.register()
        
        # Setup message subscriptions
        topic1 = f"agent:{agent1.agent_id}:messages"
        topic2 = f"agent:{agent2.agent_id}:messages"
        
        await mcp.subscribe_to_messages(topic1, agent1.message_handler)
        await mcp.subscribe_to_messages(topic2, agent2.message_handler)
        
        # Wait a moment for subscriptions to be established
        await asyncio.sleep(1)
        
        # Demonstrate message sending
        logger.info("Demonstrating message sending...")
        await agent1.send_message(
            agent2.agent_id,
            MCPMessageType.CONTEXT_UPDATE.value,
            {
                "context_type": "demo",
                "context_data": {"demo_key": "demo_value", "timestamp": datetime.utcnow().isoformat()}
            }
        )
        
        await agent2.send_message(
            agent1.agent_id,
            MCPMessageType.TASK_NOTIFICATION.value,
            {
                "task_id": "demo_task_123",
                "action": "start",
                "description": "Demo task for testing"
            }
        )
        
        # Demonstrate broadcasting
        logger.info("Demonstrating message broadcasting...")
        await agent1.broadcast_message(
            MCPMessageType.SYSTEM_STATUS.value,
            {
                "status": "operational",
                "timestamp": datetime.utcnow().isoformat(),
                "broadcast_from": agent1.agent_id
            }
        )
        
        # Wait for message processing
        await asyncio.sleep(2)
        
        # Get system status
        logger.info("Getting system status...")
        status = await mcp.get_system_status()
        logger.info(f"System status: {status}")
        
        # Get health check
        health = await mcp.get_health_check()
        logger.info(f"Health check: {health}")
        
        # Cleanup
        logger.info("Cleaning up...")
        await agent1.unregister()
        await agent2.unregister()
        
    except Exception as e:
        logger.error(f"Error in demo: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Shutdown MCP system
        await mcp.shutdown()
        logger.info("MCP system shutdown complete")


async def demo_error_handling():
    """Demonstrate error handling and recovery"""
    logger.info("=== Demo: Error Handling and Recovery ===")
    
    mcp = MCPIntegration()
    
    try:
        await mcp.initialize()
        
        # Create agent
        agent = DemoAgent("error_test_agent", mcp)
        await agent.register()
        
        # Demonstrate error handling by sending invalid message
        logger.info("Testing error handling with invalid message...")
        
        invalid_message = MCPMessage(
            id="",  # Invalid empty ID
            type="invalid_type",  # Invalid type
            source_agent="",  # Invalid empty source
            target_agents=[],  # Invalid empty targets
            payload="not_a_dict",  # Invalid payload
            timestamp=datetime.utcnow()
        )
        
        # This should trigger error handling
        success = await mcp.send_message(invalid_message)
        logger.info(f"Invalid message send result: {success}")
        
        # Get error statistics
        error_stats = mcp.error_handler.get_error_statistics()
        logger.info(f"Error statistics: {error_stats}")
        
        # Generate error report
        error_report = await mcp.error_handler.generate_error_report(hours=1)
        logger.info(f"Error report summary: {error_report.get('summary', {})}")
        
        await agent.unregister()
        
    except Exception as e:
        logger.error(f"Error in error handling demo: {e}")
    
    finally:
        await mcp.shutdown()


async def demo_message_routing():
    """Demonstrate advanced message routing"""
    logger.info("=== Demo: Advanced Message Routing ===")
    
    mcp = MCPIntegration()
    
    try:
        await mcp.initialize()
        
        # Create specialized agents
        context_agent = DemoAgent("context_processor", mcp)
        task_agent = DemoAgent("task_manager", mcp)
        logger_agent = DemoAgent("message_logger", mcp)
        
        await context_agent.register()
        await task_agent.register()
        await logger_agent.register()
        
        # Add custom routing rules
        logger.info("Adding custom routing rules...")
        
        # Route all context updates to context processor and logger
        context_rule = MessageRoutingRule(
            message_type=MCPMessageType.CONTEXT_UPDATE.value,
            route_to=["context_processor", "message_logger"]
        )
        mcp.add_message_routing_rule(context_rule)
        
        # Route all task notifications to task manager and logger
        task_rule = MessageRoutingRule(
            message_type=MCPMessageType.TASK_NOTIFICATION.value,
            route_to=["task_manager", "message_logger"]
        )
        mcp.add_message_routing_rule(task_rule)
        
        # Setup subscriptions
        await mcp.subscribe_to_messages(f"agent:{context_agent.agent_id}:messages", context_agent.message_handler)
        await mcp.subscribe_to_messages(f"agent:{task_agent.agent_id}:messages", task_agent.message_handler)
        await mcp.subscribe_to_messages(f"agent:{logger_agent.agent_id}:messages", logger_agent.message_handler)
        
        await asyncio.sleep(1)
        
        # Send messages that will be routed by rules
        logger.info("Sending messages with custom routing...")
        
        # This should be routed to context_processor and message_logger
        context_message = MCPMessage(
            id=str(uuid.uuid4()),
            type=MCPMessageType.CONTEXT_UPDATE.value,
            source_agent="external_source",
            target_agents=["original_target"],  # Will be expanded by routing rules
            payload={
                "context_type": "user_session",
                "context_data": {"user_id": "user123", "session_data": {"key": "value"}}
            },
            timestamp=datetime.utcnow()
        )
        
        await mcp.send_message(context_message)
        
        # This should be routed to task_manager and message_logger
        task_message = MCPMessage(
            id=str(uuid.uuid4()),
            type=MCPMessageType.TASK_NOTIFICATION.value,
            source_agent="task_scheduler",
            target_agents=["original_target"],  # Will be expanded by routing rules
            payload={
                "task_id": "routing_demo_task",
                "action": "create",
                "priority": "high",
                "description": "Demonstrate routing functionality"
            },
            timestamp=datetime.utcnow()
        )
        
        await mcp.send_message(task_message)
        
        # Wait for message processing
        await asyncio.sleep(2)
        
        # Check message counts
        logger.info(f"Context agent received {context_agent.message_count} messages")
        logger.info(f"Task agent received {task_agent.message_count} messages")
        logger.info(f"Logger agent received {logger_agent.message_count} messages")
        
        # Cleanup
        await context_agent.unregister()
        await task_agent.unregister()
        await logger_agent.unregister()
        
    except Exception as e:
        logger.error(f"Error in routing demo: {e}")
    
    finally:
        await mcp.shutdown()


async def main():
    """Run all demos"""
    logger.info("Starting MCP Server Infrastructure Demo")
    
    try:
        # Run demos
        await demo_basic_functionality()
        await asyncio.sleep(1)
        
        await demo_error_handling()
        await asyncio.sleep(1)
        
        await demo_message_routing()
        
        logger.info("All demos completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Check if Redis is available
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, socket_timeout=1)
        r.ping()
        logger.info("Redis connection verified")
    except Exception as e:
        logger.warning(f"Redis not available: {e}")
        logger.warning("Demo will use mocked Redis functionality")
    
    # Run the demo
    asyncio.run(main())