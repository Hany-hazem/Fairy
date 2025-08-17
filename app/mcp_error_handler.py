# app/mcp_error_handler.py
"""
MCP Error Handling and Recovery System

This module provides comprehensive error handling and recovery mechanisms
for MCP operations including exponential backoff, graceful degradation,
and comprehensive error logging and reporting.
"""

import asyncio
import json
import logging
import time
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import redis.asyncio as redis
from redis.exceptions import ConnectionError, TimeoutError, RedisError, ResponseError

from .config import settings
from .mcp_models import MCPMessage, MCPMessageType

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification"""
    CONNECTION = "connection"
    AUTHENTICATION = "authentication"
    VALIDATION = "validation"
    TIMEOUT = "timeout"
    RESOURCE = "resource"
    PROTOCOL = "protocol"
    SYSTEM = "system"
    UNKNOWN = "unknown"


class RecoveryAction(Enum):
    """Recovery actions that can be taken"""
    RETRY = "retry"
    RECONNECT = "reconnect"
    FALLBACK = "fallback"
    ESCALATE = "escalate"
    IGNORE = "ignore"
    SHUTDOWN = "shutdown"


@dataclass
class ErrorInfo:
    """Comprehensive error information"""
    error_id: str
    timestamp: datetime
    error_type: str
    error_message: str
    severity: ErrorSeverity
    category: ErrorCategory
    context: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None
    recovery_actions: List[RecoveryAction] = field(default_factory=list)
    retry_count: int = 0
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BackoffConfig:
    """Exponential backoff configuration"""
    initial_delay: float = 1.0
    max_delay: float = 300.0  # 5 minutes
    multiplier: float = 2.0
    jitter: bool = True
    max_retries: int = 10


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5
    recovery_timeout: int = 60  # seconds
    half_open_max_calls: int = 3
    success_threshold: int = 2


class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Circuit breaker for preventing cascading failures"""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.half_open_calls = 0
    
    def can_execute(self) -> bool:
        """Check if operation can be executed"""
        if self.state == CircuitBreakerState.CLOSED:
            return True
        
        if self.state == CircuitBreakerState.OPEN:
            if self.last_failure_time:
                time_since_failure = (datetime.utcnow() - self.last_failure_time).total_seconds()
                if time_since_failure >= self.config.recovery_timeout:
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.half_open_calls = 0
                    return True
            return False
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            return self.half_open_calls < self.config.half_open_max_calls
        
        return False
    
    def record_success(self):
        """Record successful operation"""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                self.success_count = 0
        else:
            self.failure_count = 0
    
    def record_failure(self):
        """Record failed operation"""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        
        if self.state == CircuitBreakerState.CLOSED:
            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitBreakerState.OPEN
        elif self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.OPEN
            self.half_open_calls = 0
            self.success_count = 0
    
    def get_state_info(self) -> Dict[str, Any]:
        """Get circuit breaker state information"""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "half_open_calls": self.half_open_calls,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None
        }


class ExponentialBackoff:
    """Exponential backoff implementation with jitter"""
    
    def __init__(self, config: BackoffConfig):
        self.config = config
        self.attempt = 0
    
    def get_delay(self) -> float:
        """Get delay for current attempt"""
        if self.attempt >= self.config.max_retries:
            return -1  # No more retries
        
        delay = min(
            self.config.initial_delay * (self.config.multiplier ** self.attempt),
            self.config.max_delay
        )
        
        # Add jitter to prevent thundering herd
        if self.config.jitter:
            import random
            delay = delay * (0.5 + random.random() * 0.5)
        
        return delay
    
    def next_attempt(self) -> float:
        """Move to next attempt and return delay"""
        delay = self.get_delay()
        self.attempt += 1
        return delay
    
    def reset(self):
        """Reset backoff state"""
        self.attempt = 0


class MCPErrorHandler:
    """
    Comprehensive MCP error handling and recovery system
    
    Features:
    - Exponential backoff for connection failures
    - Circuit breaker pattern for preventing cascading failures
    - Graceful degradation for Redis unavailability
    - Comprehensive error logging and reporting
    - Automatic recovery mechanisms
    """
    
    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or settings.REDIS_URL
        
        # Error tracking
        self._errors: Dict[str, ErrorInfo] = {}
        self._error_counts: Dict[str, int] = {}
        self._error_patterns: Dict[str, List[datetime]] = {}
        
        # Recovery mechanisms
        self._backoff_configs: Dict[str, BackoffConfig] = {}
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._recovery_handlers: Dict[str, Callable] = {}
        
        # Degradation state
        self._degradation_mode = False
        self._degraded_services: Set[str] = set()
        self._fallback_handlers: Dict[str, Callable] = {}
        
        # Redis connection for error logging
        self._redis: Optional[redis.Redis] = None
        self._redis_available = False
        
        # Background tasks
        self._background_tasks: Set[asyncio.Task] = set()
        self._running = False
        
        # Statistics
        self.stats = {
            "total_errors": 0,
            "resolved_errors": 0,
            "active_errors": 0,
            "recovery_attempts": 0,
            "successful_recoveries": 0,
            "degradation_events": 0
        }
        
        # Initialize default configurations
        self._initialize_default_configs()
        
        logger.info("MCP Error Handler initialized")
    
    def _initialize_default_configs(self):
        """Initialize default error handling configurations"""
        # Default backoff configurations
        self._backoff_configs.update({
            "redis_connection": BackoffConfig(
                initial_delay=1.0,
                max_delay=60.0,
                multiplier=2.0,
                max_retries=10
            ),
            "message_delivery": BackoffConfig(
                initial_delay=0.5,
                max_delay=30.0,
                multiplier=1.5,
                max_retries=5
            ),
            "agent_communication": BackoffConfig(
                initial_delay=2.0,
                max_delay=120.0,
                multiplier=2.0,
                max_retries=8
            )
        })
        
        # Default circuit breaker configurations
        self._circuit_breakers.update({
            "redis_operations": CircuitBreaker(CircuitBreakerConfig(
                failure_threshold=5,
                recovery_timeout=30,
                half_open_max_calls=3,
                success_threshold=2
            )),
            "message_routing": CircuitBreaker(CircuitBreakerConfig(
                failure_threshold=10,
                recovery_timeout=60,
                half_open_max_calls=5,
                success_threshold=3
            ))
        })
    
    async def start(self) -> bool:
        """Start the error handler"""
        try:
            self._running = True
            
            # Initialize Redis connection for error logging
            await self._initialize_redis_logging()
            
            # Start background tasks
            await self._start_background_tasks()
            
            logger.info("MCP Error Handler started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start MCP Error Handler: {e}")
            return False
    
    async def stop(self):
        """Stop the error handler"""
        try:
            self._running = False
            
            # Cancel background tasks
            for task in self._background_tasks:
                task.cancel()
            
            if self._background_tasks:
                await asyncio.gather(*self._background_tasks, return_exceptions=True)
            
            # Close Redis connection
            if self._redis:
                await self._redis.close()
            
            logger.info("MCP Error Handler stopped")
            
        except Exception as e:
            logger.error(f"Error stopping MCP Error Handler: {e}")
    
    async def handle_error(self, error: Exception, context: Dict[str, Any] = None, operation: str = None) -> ErrorInfo:
        """
        Handle an error with comprehensive logging and recovery
        
        Args:
            error: The exception that occurred
            context: Additional context information
            operation: The operation that failed
            
        Returns:
            ErrorInfo object with error details and recovery actions
        """
        try:
            # Create error info
            error_info = self._create_error_info(error, context, operation)
            
            # Store error
            self._errors[error_info.error_id] = error_info
            self.stats["total_errors"] += 1
            self.stats["active_errors"] += 1
            
            # Update error patterns
            self._update_error_patterns(error_info)
            
            # Log error
            await self._log_error(error_info)
            
            # Determine recovery actions
            recovery_actions = self._determine_recovery_actions(error_info)
            error_info.recovery_actions = recovery_actions
            
            # Execute recovery actions
            await self._execute_recovery_actions(error_info)
            
            # Check for degradation triggers
            await self._check_degradation_triggers(error_info)
            
            logger.error(f"Handled error {error_info.error_id}: {error_info.error_message}")
            return error_info
            
        except Exception as e:
            logger.critical(f"Error in error handler: {e}")
            # Return basic error info even if handling fails
            return ErrorInfo(
                error_id=f"error_handler_failure_{int(time.time())}",
                timestamp=datetime.utcnow(),
                error_type=type(error).__name__,
                error_message=str(error),
                severity=ErrorSeverity.CRITICAL,
                category=ErrorCategory.SYSTEM
            )
    
    async def handle_connection_error(self, error: Exception, connection_type: str = "redis") -> bool:
        """
        Handle connection errors with exponential backoff
        
        Args:
            error: Connection error
            connection_type: Type of connection (redis, agent, etc.)
            
        Returns:
            True if recovery should be attempted
        """
        try:
            # Get or create backoff handler
            backoff_key = f"{connection_type}_connection"
            if backoff_key not in self._backoff_configs:
                self._backoff_configs[backoff_key] = BackoffConfig()
            
            backoff = ExponentialBackoff(self._backoff_configs[backoff_key])
            
            # Check circuit breaker
            circuit_breaker_key = f"{connection_type}_operations"
            if circuit_breaker_key in self._circuit_breakers:
                circuit_breaker = self._circuit_breakers[circuit_breaker_key]
                if not circuit_breaker.can_execute():
                    logger.warning(f"Circuit breaker open for {connection_type}, skipping retry")
                    return False
            
            # Handle the error
            error_info = await self.handle_error(
                error,
                context={"connection_type": connection_type},
                operation=f"{connection_type}_connection"
            )
            
            # Get retry delay
            delay = backoff.next_attempt()
            if delay < 0:
                logger.error(f"Max retries exceeded for {connection_type} connection")
                return False
            
            logger.info(f"Will retry {connection_type} connection in {delay:.2f} seconds")
            await asyncio.sleep(delay)
            
            return True
            
        except Exception as e:
            logger.error(f"Error handling connection error: {e}")
            return False
    
    async def handle_redis_unavailable(self, operation: str, fallback_data: Any = None) -> Any:
        """
        Handle Redis unavailability with graceful degradation
        
        Args:
            operation: The operation that failed
            fallback_data: Optional fallback data to use
            
        Returns:
            Fallback result or None
        """
        try:
            # Enable degradation mode
            if not self._degradation_mode:
                await self._enable_degradation_mode()
            
            # Add Redis to degraded services
            self._degraded_services.add("redis")
            
            # Log degradation event
            logger.warning(f"Redis unavailable for operation {operation}, using fallback")
            self.stats["degradation_events"] += 1
            
            # Check for fallback handler
            if operation in self._fallback_handlers:
                fallback_handler = self._fallback_handlers[operation]
                return await fallback_handler(fallback_data)
            
            # Default fallback behaviors
            if operation == "publish_message":
                # Store message locally for later delivery
                return await self._store_message_locally(fallback_data)
            
            elif operation == "get_message_history":
                # Return empty history
                return []
            
            elif operation == "subscribe_to_topic":
                # Return mock subscription ID
                return f"fallback_subscription_{int(time.time())}"
            
            elif operation == "cleanup_expired_messages":
                # Skip cleanup
                return 0
            
            # Default: return None for unknown operations
            return None
            
        except Exception as e:
            logger.error(f"Error in Redis fallback handling: {e}")
            return None
    
    def register_recovery_handler(self, error_type: str, handler: Callable):
        """Register a custom recovery handler for specific error types"""
        self._recovery_handlers[error_type] = handler
        logger.info(f"Registered recovery handler for error type: {error_type}")
    
    def register_fallback_handler(self, operation: str, handler: Callable):
        """Register a fallback handler for specific operations"""
        self._fallback_handlers[operation] = handler
        logger.info(f"Registered fallback handler for operation: {operation}")
    
    def configure_backoff(self, operation: str, config: BackoffConfig):
        """Configure exponential backoff for specific operations"""
        self._backoff_configs[operation] = config
        logger.info(f"Configured backoff for operation: {operation}")
    
    def configure_circuit_breaker(self, service: str, config: CircuitBreakerConfig):
        """Configure circuit breaker for specific services"""
        self._circuit_breakers[service] = CircuitBreaker(config)
        logger.info(f"Configured circuit breaker for service: {service}")
    
    async def get_error_report(self, include_resolved: bool = False) -> Dict[str, Any]:
        """
        Get comprehensive error report
        
        Args:
            include_resolved: Whether to include resolved errors
            
        Returns:
            Error report dictionary
        """
        try:
            # Filter errors
            errors_to_include = []
            for error_info in self._errors.values():
                if include_resolved or not error_info.resolved:
                    errors_to_include.append({
                        "error_id": error_info.error_id,
                        "timestamp": error_info.timestamp.isoformat(),
                        "error_type": error_info.error_type,
                        "error_message": error_info.error_message,
                        "severity": error_info.severity.value,
                        "category": error_info.category.value,
                        "retry_count": error_info.retry_count,
                        "resolved": error_info.resolved,
                        "resolution_time": error_info.resolution_time.isoformat() if error_info.resolution_time else None,
                        "recovery_actions": [action.value for action in error_info.recovery_actions]
                    })
            
            # Get circuit breaker states
            circuit_breaker_states = {}
            for service, breaker in self._circuit_breakers.items():
                circuit_breaker_states[service] = breaker.get_state_info()
            
            # Compile report
            report = {
                "timestamp": datetime.utcnow().isoformat(),
                "statistics": self.stats.copy(),
                "degradation_mode": self._degradation_mode,
                "degraded_services": list(self._degraded_services),
                "errors": errors_to_include,
                "error_patterns": self._get_error_pattern_summary(),
                "circuit_breakers": circuit_breaker_states,
                "redis_available": self._redis_available
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating error report: {e}")
            return {"error": f"Failed to generate report: {e}"}
    
    async def resolve_error(self, error_id: str, resolution_notes: str = None) -> bool:
        """
        Mark an error as resolved
        
        Args:
            error_id: Error ID to resolve
            resolution_notes: Optional resolution notes
            
        Returns:
            True if error was resolved successfully
        """
        try:
            if error_id not in self._errors:
                logger.warning(f"Error {error_id} not found")
                return False
            
            error_info = self._errors[error_id]
            error_info.resolved = True
            error_info.resolution_time = datetime.utcnow()
            
            if resolution_notes:
                error_info.metadata["resolution_notes"] = resolution_notes
            
            # Update statistics
            self.stats["resolved_errors"] += 1
            self.stats["active_errors"] = max(0, self.stats["active_errors"] - 1)
            
            # Log resolution
            await self._log_error_resolution(error_info, resolution_notes)
            
            logger.info(f"Resolved error {error_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error resolving error {error_id}: {e}")
            return False
    
    def _create_error_info(self, error: Exception, context: Dict[str, Any] = None, operation: str = None) -> ErrorInfo:
        """Create comprehensive error information"""
        error_id = f"error_{int(time.time() * 1000)}_{id(error)}"
        
        # Determine error category
        category = self._categorize_error(error)
        
        # Determine error severity
        severity = self._determine_severity(error, category)
        
        # Create error info
        error_info = ErrorInfo(
            error_id=error_id,
            timestamp=datetime.utcnow(),
            error_type=type(error).__name__,
            error_message=str(error),
            severity=severity,
            category=category,
            context=context or {},
            stack_trace=traceback.format_exc()
        )
        
        # Add operation context
        if operation:
            error_info.context["operation"] = operation
        
        return error_info
    
    def _categorize_error(self, error: Exception) -> ErrorCategory:
        """Categorize error based on type and message"""
        error_type = type(error).__name__
        error_message = str(error).lower()
        
        # Connection errors
        if isinstance(error, (ConnectionError, TimeoutError)) or "connection" in error_message:
            return ErrorCategory.CONNECTION
        
        # Authentication errors
        if "auth" in error_message or "permission" in error_message:
            return ErrorCategory.AUTHENTICATION
        
        # Validation errors
        if "validation" in error_message or "invalid" in error_message:
            return ErrorCategory.VALIDATION
        
        # Timeout errors
        if "timeout" in error_message or isinstance(error, asyncio.TimeoutError):
            return ErrorCategory.TIMEOUT
        
        # Resource errors
        if "memory" in error_message or "disk" in error_message or "resource" in error_message:
            return ErrorCategory.RESOURCE
        
        # Protocol errors
        if isinstance(error, ResponseError) or "protocol" in error_message:
            return ErrorCategory.PROTOCOL
        
        # System errors
        if isinstance(error, (OSError, SystemError)):
            return ErrorCategory.SYSTEM
        
        return ErrorCategory.UNKNOWN
    
    def _determine_severity(self, error: Exception, category: ErrorCategory) -> ErrorSeverity:
        """Determine error severity"""
        # Critical errors
        if category in [ErrorCategory.SYSTEM, ErrorCategory.RESOURCE]:
            return ErrorSeverity.CRITICAL
        
        # High severity errors
        if category in [ErrorCategory.CONNECTION, ErrorCategory.AUTHENTICATION]:
            return ErrorSeverity.HIGH
        
        # Medium severity errors
        if category in [ErrorCategory.TIMEOUT, ErrorCategory.PROTOCOL]:
            return ErrorSeverity.MEDIUM
        
        # Low severity errors
        return ErrorSeverity.LOW
    
    def _determine_recovery_actions(self, error_info: ErrorInfo) -> List[RecoveryAction]:
        """Determine appropriate recovery actions for an error"""
        actions = []
        
        # Based on category
        if error_info.category == ErrorCategory.CONNECTION:
            actions.extend([RecoveryAction.RETRY, RecoveryAction.RECONNECT])
        
        elif error_info.category == ErrorCategory.TIMEOUT:
            actions.extend([RecoveryAction.RETRY, RecoveryAction.FALLBACK])
        
        elif error_info.category == ErrorCategory.VALIDATION:
            actions.append(RecoveryAction.ESCALATE)
        
        elif error_info.category == ErrorCategory.RESOURCE:
            actions.extend([RecoveryAction.FALLBACK, RecoveryAction.ESCALATE])
        
        elif error_info.category == ErrorCategory.AUTHENTICATION:
            actions.extend([RecoveryAction.RECONNECT, RecoveryAction.ESCALATE])
        
        # Based on severity
        if error_info.severity == ErrorSeverity.CRITICAL:
            if RecoveryAction.ESCALATE not in actions:
                actions.append(RecoveryAction.ESCALATE)
        
        elif error_info.severity == ErrorSeverity.LOW:
            if not actions:
                actions.append(RecoveryAction.IGNORE)
        
        # Default action
        if not actions:
            actions.append(RecoveryAction.RETRY)
        
        return actions
    
    async def _execute_recovery_actions(self, error_info: ErrorInfo):
        """Execute recovery actions for an error"""
        try:
            for action in error_info.recovery_actions:
                if action == RecoveryAction.RETRY:
                    # Retry is handled by the calling code
                    pass
                
                elif action == RecoveryAction.RECONNECT:
                    # Trigger reconnection
                    await self._trigger_reconnection(error_info)
                
                elif action == RecoveryAction.FALLBACK:
                    # Enable fallback mode
                    await self._enable_fallback_mode(error_info)
                
                elif action == RecoveryAction.ESCALATE:
                    # Escalate error
                    await self._escalate_error(error_info)
                
                elif action == RecoveryAction.IGNORE:
                    # Mark as resolved
                    error_info.resolved = True
                    error_info.resolution_time = datetime.utcnow()
                
                elif action == RecoveryAction.SHUTDOWN:
                    # Trigger graceful shutdown
                    await self._trigger_shutdown(error_info)
            
            self.stats["recovery_attempts"] += 1
            
        except Exception as e:
            logger.error(f"Error executing recovery actions: {e}")
    
    async def _trigger_reconnection(self, error_info: ErrorInfo):
        """Trigger reconnection for connection errors"""
        try:
            operation = error_info.context.get("operation", "unknown")
            logger.info(f"Triggering reconnection for operation: {operation}")
            
            # This would trigger reconnection in the calling service
            # For now, we just log the action
            
        except Exception as e:
            logger.error(f"Error triggering reconnection: {e}")
    
    async def _enable_fallback_mode(self, error_info: ErrorInfo):
        """Enable fallback mode for the affected service"""
        try:
            operation = error_info.context.get("operation", "unknown")
            logger.info(f"Enabling fallback mode for operation: {operation}")
            
            if not self._degradation_mode:
                await self._enable_degradation_mode()
            
        except Exception as e:
            logger.error(f"Error enabling fallback mode: {e}")
    
    async def _escalate_error(self, error_info: ErrorInfo):
        """Escalate error to higher-level systems"""
        try:
            logger.critical(f"Escalating error {error_info.error_id}: {error_info.error_message}")
            
            # This would integrate with monitoring/alerting systems
            # For now, we just log at critical level
            
        except Exception as e:
            logger.error(f"Error escalating error: {e}")
    
    async def _trigger_shutdown(self, error_info: ErrorInfo):
        """Trigger graceful shutdown for critical errors"""
        try:
            logger.critical(f"Triggering shutdown due to critical error {error_info.error_id}")
            
            # This would trigger graceful shutdown of the service
            # For now, we just log the action
            
        except Exception as e:
            logger.error(f"Error triggering shutdown: {e}")
    
    def _update_error_patterns(self, error_info: ErrorInfo):
        """Update error patterns for trend analysis"""
        try:
            pattern_key = f"{error_info.error_type}:{error_info.category.value}"
            
            if pattern_key not in self._error_patterns:
                self._error_patterns[pattern_key] = []
            
            self._error_patterns[pattern_key].append(error_info.timestamp)
            
            # Keep only recent errors (last 24 hours)
            cutoff_time = datetime.utcnow() - timedelta(hours=24)
            self._error_patterns[pattern_key] = [
                timestamp for timestamp in self._error_patterns[pattern_key]
                if timestamp > cutoff_time
            ]
            
            # Update error counts
            if pattern_key not in self._error_counts:
                self._error_counts[pattern_key] = 0
            self._error_counts[pattern_key] += 1
            
        except Exception as e:
            logger.error(f"Error updating error patterns: {e}")
    
    async def _check_degradation_triggers(self, error_info: ErrorInfo):
        """Check if error should trigger degradation mode"""
        try:
            pattern_key = f"{error_info.error_type}:{error_info.category.value}"
            
            # Check error frequency
            if pattern_key in self._error_patterns:
                recent_errors = len(self._error_patterns[pattern_key])
                
                # Trigger degradation if too many errors in short time
                if recent_errors >= 5:  # 5 errors in 24 hours
                    logger.warning(f"High error frequency detected for {pattern_key}, enabling degradation mode")
                    await self._enable_degradation_mode()
            
        except Exception as e:
            logger.error(f"Error checking degradation triggers: {e}")
    
    async def _enable_degradation_mode(self):
        """Enable degradation mode"""
        try:
            if not self._degradation_mode:
                self._degradation_mode = True
                self.stats["degradation_events"] += 1
                
                logger.warning("Degradation mode enabled")
                
                # Notify other components about degradation mode
                # This would integrate with the main MCP system
                
        except Exception as e:
            logger.error(f"Error enabling degradation mode: {e}")
    
    async def _store_message_locally(self, message_data: Any) -> str:
        """Store message locally when Redis is unavailable"""
        try:
            # This would implement local storage for messages
            # For now, we just return a mock ID
            local_id = f"local_message_{int(time.time())}"
            logger.debug(f"Stored message locally with ID: {local_id}")
            return local_id
            
        except Exception as e:
            logger.error(f"Error storing message locally: {e}")
            return None
    
    def _get_error_pattern_summary(self) -> Dict[str, Any]:
        """Get summary of error patterns"""
        try:
            summary = {}
            
            for pattern_key, timestamps in self._error_patterns.items():
                summary[pattern_key] = {
                    "count": len(timestamps),
                    "first_occurrence": min(timestamps).isoformat() if timestamps else None,
                    "last_occurrence": max(timestamps).isoformat() if timestamps else None,
                    "frequency": len(timestamps) / 24.0  # errors per hour over 24 hours
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating error pattern summary: {e}")
            return {}
    
    async def _initialize_redis_logging(self):
        """Initialize Redis connection for error logging"""
        try:
            self._redis = redis.from_url(self.redis_url, decode_responses=True)
            await self._redis.ping()
            self._redis_available = True
            logger.info("Redis connection established for error logging")
            
        except Exception as e:
            logger.warning(f"Redis not available for error logging: {e}")
            self._redis_available = False
    
    async def _log_error(self, error_info: ErrorInfo):
        """Log error to Redis and local logs"""
        try:
            # Always log locally
            log_level = {
                ErrorSeverity.LOW: logging.INFO,
                ErrorSeverity.MEDIUM: logging.WARNING,
                ErrorSeverity.HIGH: logging.ERROR,
                ErrorSeverity.CRITICAL: logging.CRITICAL
            }.get(error_info.severity, logging.ERROR)
            
            logger.log(log_level, f"MCP Error [{error_info.error_id}]: {error_info.error_message}")
            
            # Log to Redis if available
            if self._redis_available and self._redis:
                error_data = {
                    "error_id": error_info.error_id,
                    "timestamp": error_info.timestamp.isoformat(),
                    "error_type": error_info.error_type,
                    "error_message": error_info.error_message,
                    "severity": error_info.severity.value,
                    "category": error_info.category.value,
                    "context": error_info.context,
                    "stack_trace": error_info.stack_trace
                }
                
                # Store in Redis with TTL
                await self._redis.setex(
                    f"mcp:errors:{error_info.error_id}",
                    86400,  # 24 hours TTL
                    json.dumps(error_data)
                )
                
                # Add to error index
                await self._redis.zadd(
                    "mcp:error_index",
                    {error_info.error_id: error_info.timestamp.timestamp()}
                )
            
        except Exception as e:
            logger.error(f"Error logging error information: {e}")
    
    async def _log_error_resolution(self, error_info: ErrorInfo, resolution_notes: str = None):
        """Log error resolution"""
        try:
            logger.info(f"Error resolved [{error_info.error_id}]: {resolution_notes or 'No notes'}")
            
            # Update Redis if available
            if self._redis_available and self._redis:
                resolution_data = {
                    "resolved": True,
                    "resolution_time": error_info.resolution_time.isoformat(),
                    "resolution_notes": resolution_notes
                }
                
                # Update error record
                error_key = f"mcp:errors:{error_info.error_id}"
                if await self._redis.exists(error_key):
                    error_data = json.loads(await self._redis.get(error_key))
                    error_data.update(resolution_data)
                    await self._redis.setex(error_key, 86400, json.dumps(error_data))
            
        except Exception as e:
            logger.error(f"Error logging error resolution: {e}")
    
    async def _start_background_tasks(self):
        """Start background maintenance tasks"""
        try:
            # Error cleanup task
            cleanup_task = asyncio.create_task(self._error_cleanup_loop())
            self._background_tasks.add(cleanup_task)
            
            # Pattern analysis task
            analysis_task = asyncio.create_task(self._pattern_analysis_loop())
            self._background_tasks.add(analysis_task)
            
            # Health monitoring task
            health_task = asyncio.create_task(self._health_monitoring_loop())
            self._background_tasks.add(health_task)
            
            logger.info("Started error handler background tasks")
            
        except Exception as e:
            logger.error(f"Error starting background tasks: {e}")
    
    async def _error_cleanup_loop(self):
        """Clean up old resolved errors"""
        while self._running:
            try:
                current_time = datetime.utcnow()
                cutoff_time = current_time - timedelta(hours=24)
                
                # Clean up old resolved errors
                errors_to_remove = []
                for error_id, error_info in self._errors.items():
                    if (error_info.resolved and 
                        error_info.resolution_time and 
                        error_info.resolution_time < cutoff_time):
                        errors_to_remove.append(error_id)
                
                for error_id in errors_to_remove:
                    del self._errors[error_id]
                
                if errors_to_remove:
                    logger.debug(f"Cleaned up {len(errors_to_remove)} old resolved errors")
                
                await asyncio.sleep(3600)  # Run every hour
                
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(3600)
    
    async def _pattern_analysis_loop(self):
        """Analyze error patterns for trends"""
        while self._running:
            try:
                # Analyze error patterns and trigger alerts if needed
                for pattern_key, timestamps in self._error_patterns.items():
                    if len(timestamps) >= 10:  # 10 errors in 24 hours
                        logger.warning(f"High error frequency detected: {pattern_key} ({len(timestamps)} errors)")
                
                await asyncio.sleep(1800)  # Run every 30 minutes
                
            except Exception as e:
                logger.error(f"Error in pattern analysis loop: {e}")
                await asyncio.sleep(1800)
    
    async def _health_monitoring_loop(self):
        """Monitor system health and recovery status"""
        while self._running:
            try:
                # Check if we can exit degradation mode
                if self._degradation_mode:
                    # Check if Redis is back online
                    if not self._redis_available:
                        try:
                            if self._redis:
                                await self._redis.ping()
                                self._redis_available = True
                                logger.info("Redis connection restored")
                        except:
                            pass
                    
                    # Check if we can exit degradation mode
                    if self._redis_available and len(self._degraded_services) == 1 and "redis" in self._degraded_services:
                        self._degradation_mode = False
                        self._degraded_services.clear()
                        logger.info("Exited degradation mode - services restored")
                
                await asyncio.sleep(60)  # Run every minute
                
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(60)


# Global error handler instance
mcp_error_handler = MCPErrorHandler()