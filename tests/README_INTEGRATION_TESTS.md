# Comprehensive Integration Tests

This directory contains comprehensive end-to-end integration tests for the Self-Improving AI Assistant system, covering both conversation management and self-improvement cycles.

## Test Coverage

### 1. End-to-End Conversation Integration Tests (`test_e2e_conversation_integration.py`)

Tests the complete conversation flow with real LM Studio integration, multi-session management, context persistence, and performance benchmarking.

**Requirements Covered:** 1.1, 1.2, 1.4, 2.1, 2.2, 2.3, 2.4

#### Test Classes:

- **TestE2EConversationIntegration**: Main conversation flow tests
- **TestE2EConversationWithRealLMStudio**: Integration tests with actual LM Studio (requires running instance)

#### Key Test Scenarios:

1. **Single Conversation Flow**
   - Complete conversation with context management
   - Message history tracking
   - Session persistence

2. **Multi-Session Conversation Management**
   - Concurrent sessions for different users
   - Session isolation verification
   - Cross-contamination prevention

3. **Context Persistence and Retrieval**
   - Long-term context storage
   - Vector-based context retrieval
   - Context summarization for long conversations

4. **Performance Benchmarking**
   - Response time measurement
   - Context length optimization
   - Memory usage tracking

5. **Error Handling and Recovery**
   - LLM connection failures
   - Safety filter integration
   - Graceful degradation

6. **Concurrent Session Handling**
   - Multiple simultaneous conversations
   - Resource management
   - Performance under load

### 2. End-to-End Self-Improvement Integration Tests (`test_e2e_self_improvement_integration.py`)

Tests complete self-improvement workflows in isolated environments, including safety mechanisms, rollback scenarios, and performance validation.

**Requirements Covered:** 3.1, 3.2, 3.3, 3.4, 4.1, 4.2, 4.3, 4.4, 5.1, 5.2, 5.3, 5.4, 5.5

#### Test Classes:

- **TestE2ESelfImprovementIntegration**: Main self-improvement cycle tests
- **TestSelfImprovementSafetyMechanisms**: Safety mechanism validation
- **TestSelfImprovementPerformanceValidation**: Performance improvement validation

#### Key Test Scenarios:

1. **Complete Self-Improvement Cycle**
   - Analysis → Planning → Testing → Application → Validation
   - File modification with rollback points
   - Success verification

2. **Safety Mechanism Validation**
   - Risk level filtering (Conservative/Moderate/Aggressive)
   - Dangerous code change prevention
   - Safety threshold enforcement

3. **Rollback on Test Failure**
   - Automatic rollback when tests fail
   - Git integration for version control
   - Integrity verification

4. **Performance Improvement Validation**
   - Before/after performance comparison
   - Regression detection
   - Improvement measurement

5. **Concurrent Cycle Prevention**
   - Single active cycle enforcement
   - Queue management
   - Resource locking

6. **Emergency Stop Functionality**
   - Immediate cycle termination
   - Safe state restoration
   - Cleanup procedures

7. **Multi-File Improvements**
   - Cross-file dependency handling
   - Atomic transaction support
   - Consistency verification

8. **Status Tracking**
   - Real-time status monitoring
   - Progress reporting
   - Error state handling

## Running the Tests

### Prerequisites

1. **Python Environment**: Python 3.8+ with required dependencies
2. **Redis**: Running Redis instance for session management
3. **Git**: Initialized git repository for version control tests
4. **LM Studio** (optional): For real LLM integration tests

### Quick Start

Run all integration tests:
```bash
python run_integration_tests.py
```

Run specific test suites:
```bash
# Conversation tests only
python -m pytest tests/test_e2e_conversation_integration.py -v

# Self-improvement tests only
python -m pytest tests/test_e2e_self_improvement_integration.py -v

# Specific test
python -m pytest tests/test_e2e_conversation_integration.py::TestE2EConversationIntegration::test_single_conversation_flow -v
```

### Test Configuration

Tests use isolated environments and mocked dependencies by default:

- **Isolated Project Environments**: Each test creates temporary directories
- **Mocked LLM Responses**: Realistic but predictable responses
- **In-Memory Storage**: Redis fallback for CI/CD environments
- **Git Integration**: Temporary repositories for version control tests

### Real Integration Testing

For tests with actual external services:

```bash
# Run with real LM Studio (requires running instance)
python -m pytest tests/test_e2e_conversation_integration.py -m integration -v

# Skip integration tests
python -m pytest tests/test_e2e_conversation_integration.py -m "not integration" -v
```

## Test Architecture

### Fixtures and Mocking

- **Isolated Environments**: Temporary directories with full project structure
- **Mock LLM Clients**: Realistic response patterns without external dependencies
- **Performance Monitors**: Simulated metrics for consistent testing
- **Git Repositories**: Temporary repos for version control testing

### Safety Testing

- **Sandboxed Execution**: All tests run in isolated environments
- **No External Modifications**: Tests don't affect the main codebase
- **Automatic Cleanup**: Temporary resources are cleaned up after tests
- **Error Isolation**: Test failures don't impact other tests

### Performance Testing

- **Benchmarking**: Response time and resource usage measurement
- **Load Testing**: Concurrent session handling
- **Memory Profiling**: Memory usage tracking and leak detection
- **Regression Testing**: Performance comparison over time

## Expected Results

### Success Criteria

- **All Tests Pass**: No failures in the test suite
- **Performance Benchmarks**: Response times within acceptable limits
- **Safety Validation**: Dangerous changes properly blocked
- **Rollback Verification**: Failed changes properly reverted
- **Context Persistence**: Conversation context maintained across sessions

### Performance Benchmarks

- **Average Response Time**: < 5 seconds per query
- **Session Creation**: < 2 seconds
- **Context Retrieval**: < 1 second
- **Safety Validation**: < 0.5 seconds
- **Rollback Operations**: < 3 seconds

## Troubleshooting

### Common Issues

1. **Redis Connection Errors**
   - Ensure Redis is running on localhost:6379
   - Tests fall back to in-memory storage if Redis unavailable

2. **Git Repository Errors**
   - Tests create temporary git repositories
   - Ensure git is installed and configured

3. **Permission Errors**
   - Tests create temporary directories
   - Ensure write permissions in test directory

4. **LM Studio Connection**
   - Real integration tests require LM Studio running on localhost:1234
   - Mock tests work without external dependencies

### Debug Mode

Run tests with detailed output:
```bash
python -m pytest tests/test_e2e_conversation_integration.py -v -s --tb=long
```

Enable logging:
```bash
python -m pytest tests/test_e2e_conversation_integration.py -v --log-cli-level=DEBUG
```

## Continuous Integration

These tests are designed for CI/CD environments:

- **No External Dependencies**: Mock all external services
- **Deterministic Results**: Consistent behavior across environments
- **Fast Execution**: Optimized for quick feedback
- **Comprehensive Coverage**: Full system validation

### CI Configuration Example

```yaml
name: Integration Tests
on: [push, pull_request]
jobs:
  integration-tests:
    runs-on: ubuntu-latest
    services:
      redis:
        image: redis:6
        ports:
          - 6379:6379
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run integration tests
        run: python run_integration_tests.py
```

## Contributing

When adding new integration tests:

1. **Follow Patterns**: Use existing test patterns and fixtures
2. **Isolated Testing**: Ensure tests don't affect each other
3. **Comprehensive Coverage**: Test both success and failure scenarios
4. **Performance Aware**: Include performance assertions
5. **Safety First**: Validate all safety mechanisms
6. **Documentation**: Update this README with new test descriptions

## Test Metrics

The integration test suite provides comprehensive metrics:

- **Functional Coverage**: All major user workflows
- **Error Coverage**: All error conditions and recovery scenarios
- **Performance Coverage**: Response times, memory usage, throughput
- **Safety Coverage**: All security and safety mechanisms
- **Integration Coverage**: All external service interactions