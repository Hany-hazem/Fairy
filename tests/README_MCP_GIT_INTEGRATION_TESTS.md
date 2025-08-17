# MCP and Git Workflow Integration Tests

This document describes the comprehensive integration tests for the MCP (Model Context Protocol) and Git workflow systems implemented as part of task 11 in the MCP-Git Integration specification.

## Overview

The integration tests validate the complete functionality of:

1. **MCP System Integration** - End-to-end communication between agents via MCP
2. **Git Workflow Integration** - Automated Git operations for task lifecycle management
3. **Performance and Reliability** - System behavior under load and error conditions

## Test Structure

### MCP Integration Tests (`test_mcp_integration_comprehensive.py`)

#### TestMCPEndToEndCommunication
Tests complete MCP communication workflows between multiple agents:

- **Agent Registration and Discovery**: Validates agent registration with capabilities and server discovery
- **Message Routing**: Tests message routing between agents with different message types
- **Context Synchronization**: Validates context sharing and synchronization between agents
- **Task Notification Workflow**: Tests task lifecycle notifications (started, progress, completed)
- **Heartbeat and Health Monitoring**: Validates agent health monitoring and heartbeat systems
- **Message Acknowledgment**: Tests message delivery confirmation and acknowledgment
- **Broadcast Functionality**: Validates broadcasting messages to multiple agents
- **Message Priority Handling**: Tests priority-based message processing

#### TestRedisBackendIntegration
Tests Redis backend functionality and failover scenarios:

- **Connection Lifecycle**: Tests Redis connection establishment and cleanup
- **Connection Failover**: Validates automatic reconnection and failover handling
- **Message Persistence**: Tests message persistence and replay capabilities
- **Queue Management**: Validates message queue management and cleanup
- **Subscription Management**: Tests pub/sub subscription lifecycle
- **Batch Operations**: Tests batch message publishing for performance
- **Memory Management**: Validates Redis memory usage and cleanup policies

#### TestContextSynchronizationValidation
Tests context synchronization and conflict resolution:

- **Successful Synchronization**: Tests normal context synchronization between agents
- **Conflict Detection and Resolution**: Validates conflict detection and resolution strategies
- **Access Control Validation**: Tests context access control and permissions
- **Handler Registration**: Tests context handler registration and execution
- **Multi-Agent Conflicts**: Tests conflict resolution between multiple agents
- **Performance Metrics**: Validates synchronization performance tracking
- **Subscription and Notification**: Tests context subscription and notification systems

#### TestMCPPerformanceAndReliability
Tests system performance and reliability under various conditions:

- **High Volume Processing**: Tests processing large numbers of messages
- **Concurrent Connections**: Validates handling multiple concurrent agent connections
- **Message Delivery Reliability**: Tests message delivery reliability and retry mechanisms
- **System Recovery**: Tests recovery after component failures
- **Memory Usage**: Validates memory usage under sustained load

### Git Workflow Integration Tests (`test_git_workflow_integration_comprehensive.py`)

#### TestTaskLifecycleGitIntegration
Tests complete task lifecycle integration with Git operations:

- **Complete Lifecycle**: Tests full task lifecycle from start to completion with Git integration
- **Concurrent Tasks**: Validates handling multiple concurrent tasks
- **Task Dependencies**: Tests dependency handling in Git workflows
- **Branch Naming**: Validates branch naming conventions for different task types
- **Commit Messages**: Tests intelligent commit message generation

#### TestMergeConflictResolution
Tests merge conflict detection and resolution:

- **Conflict Detection**: Tests detection of merge conflicts between branches
- **Resolution Strategies**: Validates different conflict resolution strategies
- **Automated Resolution**: Tests automated conflict resolution mechanisms
- **Merge Strategy Generation**: Tests generation of merge strategies for dependent tasks

#### TestGitWorkflowPerformanceAndReliability
Tests Git workflow performance and reliability:

- **High Volume Processing**: Tests processing many tasks concurrently
- **Concurrent Operations**: Validates concurrent branch operations
- **Large File Performance**: Tests performance with large file commits
- **Operation Reliability**: Tests reliability of Git operations under various conditions
- **Error Recovery**: Tests error recovery mechanisms
- **Monitoring and Health Checks**: Validates workflow monitoring and health checks
- **Metrics and Analytics**: Tests Git metrics collection and analytics

## Requirements Coverage

The tests cover all requirements specified in task 11:

### Task 11.1 Requirements (1.1, 1.2, 1.3, 2.1, 2.2, 3.1, 3.2, 7.1, 7.2)

- **1.1-1.3**: MCP server implementation and functionality
- **2.1-2.2**: MCP client integration and message handling
- **3.1-3.2**: Redis integration for MCP with persistence and pub/sub
- **7.1-7.2**: Context synchronization via MCP between agents

### Task 11.2 Requirements (4.1, 4.2, 4.3, 5.1, 5.2, 8.1, 8.2)

- **4.1-4.3**: Automated Git workflow with branch management and commits
- **5.1-5.2**: Git integration with task management and traceability
- **8.1-8.2**: Git workflow automation with monitoring and recovery

## Running the Tests

### Prerequisites

Ensure you have the required dependencies installed:

```bash
pip install pytest pytest-asyncio redis
```

### Running All Tests

Use the provided test runner:

```bash
python run_integration_tests.py
```

### Running Specific Test Suites

```bash
# MCP tests only
python run_integration_tests.py --mcp-only

# Git workflow tests only
python run_integration_tests.py --git-only

# Specific test file
python run_integration_tests.py --test-file tests/test_mcp_integration_comprehensive.py

# Quick tests (core functionality only)
python run_integration_tests.py --quick

# Verbose output
python run_integration_tests.py --verbose
```

### Running Individual Test Classes

```bash
# Run specific test class
pytest tests/test_mcp_integration_comprehensive.py::TestMCPEndToEndCommunication -v

# Run specific test method
pytest tests/test_mcp_integration_comprehensive.py::TestMCPEndToEndCommunication::test_agent_registration_and_discovery -v
```

## Test Environment Setup

### Temporary Git Repositories

The Git workflow tests use temporary Git repositories created with:
- Proper Git configuration (user.name, user.email)
- Initial commit structure
- Automatic cleanup after tests

### Mock Redis Backend

The MCP tests use mocked Redis backends to avoid requiring actual Redis instances:
- Simulated connection management
- Mocked pub/sub functionality
- Simulated message persistence

### Async Test Support

All async tests use pytest-asyncio for proper async/await support:
- Proper event loop management
- Timeout handling for long-running operations
- Concurrent operation testing

## Performance Benchmarks

The tests include performance benchmarks to ensure system scalability:

### MCP Performance Targets
- Process 100 messages in under 10 seconds
- Handle 20 concurrent agent connections
- Message delivery reliability > 95%

### Git Workflow Performance Targets
- Process 50 tasks in under 30 seconds
- Create 20 branches in under 10 seconds
- Commit 5MB of files in under 30 seconds

## Error Scenarios Tested

### MCP Error Scenarios
- Redis connection failures and recovery
- Message validation failures
- Agent disconnection and reconnection
- Context synchronization conflicts
- High memory usage conditions

### Git Workflow Error Scenarios
- Merge conflicts between branches
- Failed commit operations
- Corrupted tracking data recovery
- Concurrent operation conflicts
- Large file handling

## Continuous Integration

The tests are designed to run in CI/CD environments:

- **Timeout Protection**: All tests have reasonable timeouts
- **Resource Cleanup**: Proper cleanup of temporary resources
- **Deterministic Results**: Tests avoid race conditions and timing dependencies
- **Comprehensive Reporting**: Detailed test results and failure information

## Extending the Tests

### Adding New Test Cases

1. **MCP Tests**: Add new test methods to appropriate test classes in `test_mcp_integration_comprehensive.py`
2. **Git Tests**: Add new test methods to appropriate test classes in `test_git_workflow_integration_comprehensive.py`

### Test Fixtures

Use existing fixtures for common setup:
- `temp_git_repo`: Temporary Git repository
- `mock_redis`: Mocked Redis backend
- `mcp_clients`: Multiple MCP client instances
- `automation_service`: Git workflow automation service

### Performance Tests

When adding performance tests:
- Set reasonable performance targets
- Use appropriate timeouts
- Clean up resources properly
- Provide meaningful performance metrics

## Troubleshooting

### Common Issues

1. **Git Configuration**: Ensure Git is properly configured with user.name and user.email
2. **Async Timeouts**: Increase timeouts for slow systems
3. **Resource Cleanup**: Ensure temporary directories are properly cleaned up
4. **Mock Failures**: Verify mock objects are properly configured

### Debug Mode

Run tests with verbose output for debugging:

```bash
python run_integration_tests.py --verbose --no-capture
```

### Individual Test Debugging

```bash
pytest tests/test_mcp_integration_comprehensive.py::TestMCPEndToEndCommunication::test_agent_registration_and_discovery -v -s --tb=long
```

## Test Coverage

The integration tests provide comprehensive coverage of:

- **MCP Protocol Implementation**: All message types and routing scenarios
- **Redis Backend Integration**: Connection management, pub/sub, and persistence
- **Context Synchronization**: Conflict detection, resolution, and access control
- **Git Workflow Automation**: Task lifecycle, branch management, and commit generation
- **Performance and Reliability**: Load testing, error recovery, and monitoring
- **Error Handling**: Failure scenarios and recovery mechanisms

This comprehensive test suite ensures the reliability and performance of the MCP and Git workflow integration systems under various real-world conditions.