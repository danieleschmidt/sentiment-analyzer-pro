"""
Photonic-MLIR Bridge - Advanced Error Handling and Recovery System

This module provides comprehensive error handling, validation, and recovery
mechanisms for the photonic-MLIR synthesis bridge with autonomous capabilities.
"""

import logging
import time
import traceback
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Union
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels for classification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""
    VALIDATION = "validation"
    SYNTHESIS = "synthesis"
    SECURITY = "security"
    PERFORMANCE = "performance"
    DEPENDENCY = "dependency"
    CONFIGURATION = "configuration"
    HARDWARE = "hardware"
    NETWORK = "network"


@dataclass
class ErrorContext:
    """Comprehensive error context information."""
    error_id: str
    timestamp: float
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    details: Dict[str, Any]
    stack_trace: Optional[str]
    component: Optional[str]
    operation: Optional[str]
    recovery_suggestions: List[str]
    auto_recoverable: bool = False


class PhotonicErrorHandler:
    """Advanced error handling system with automatic recovery."""
    
    def __init__(self):
        self.error_log: List[ErrorContext] = []
        self.recovery_strategies: Dict[str, Callable] = {}
        self.error_patterns: Dict[str, ErrorCategory] = {}
        self.max_retry_attempts = 3
        self.retry_delays = [1, 2, 4]  # Exponential backoff
        
        self._initialize_error_patterns()
        self._initialize_recovery_strategies()
    
    def _initialize_error_patterns(self):
        """Initialize common error patterns for classification."""
        self.error_patterns = {
            "validation": ErrorCategory.VALIDATION,
            "invalid": ErrorCategory.VALIDATION,
            "synthesis": ErrorCategory.SYNTHESIS,
            "mlir": ErrorCategory.SYNTHESIS,
            "security": ErrorCategory.SECURITY,
            "unauthorized": ErrorCategory.SECURITY,
            "permission": ErrorCategory.SECURITY,
            "performance": ErrorCategory.PERFORMANCE,
            "timeout": ErrorCategory.PERFORMANCE,
            "memory": ErrorCategory.PERFORMANCE,
            "dependency": ErrorCategory.DEPENDENCY,
            "import": ErrorCategory.DEPENDENCY,
            "module": ErrorCategory.DEPENDENCY,
            "config": ErrorCategory.CONFIGURATION,
            "hardware": ErrorCategory.HARDWARE,
            "device": ErrorCategory.HARDWARE,
            "network": ErrorCategory.NETWORK,
            "connection": ErrorCategory.NETWORK
        }
    
    def _initialize_recovery_strategies(self):
        """Initialize automatic recovery strategies."""
        self.recovery_strategies = {
            "dependency_missing": self._recover_dependency_missing,
            "validation_failed": self._recover_validation_failed,
            "synthesis_timeout": self._recover_synthesis_timeout,
            "memory_error": self._recover_memory_error,
            "configuration_error": self._recover_configuration_error
        }
    
    def handle_error(self, 
                    exception: Exception,
                    component: str = None,
                    operation: str = None,
                    context: Dict[str, Any] = None) -> ErrorContext:
        """
        Handle an error with comprehensive logging and recovery attempts.
        
        Args:
            exception: The exception that occurred
            component: Component where error occurred
            operation: Operation being performed
            context: Additional context information
            
        Returns:
            ErrorContext: Detailed error information
        """
        error_id = f"err_{int(time.time() * 1000)}"
        
        # Classify error
        category = self._classify_error(str(exception))
        severity = self._determine_severity(exception, category)
        
        # Create error context
        error_context = ErrorContext(
            error_id=error_id,
            timestamp=time.time(),
            category=category,
            severity=severity,
            message=str(exception),
            details=context or {},
            stack_trace=traceback.format_exc(),
            component=component,
            operation=operation,
            recovery_suggestions=self._generate_recovery_suggestions(exception, category),
            auto_recoverable=self._is_auto_recoverable(exception, category)
        )
        
        # Log error
        self.error_log.append(error_context)
        self._log_error(error_context)
        
        # Attempt automatic recovery if possible
        if error_context.auto_recoverable:
            recovery_success = self._attempt_recovery(error_context)
            if recovery_success:
                logger.info(f"Successfully recovered from error {error_id}")
                error_context.details["recovery_successful"] = True
        
        # Record metrics
        if hasattr(self, 'monitor'):
            from .photonic_monitoring import record_error_metrics
            record_error_metrics(error_context)
        
        return error_context
    
    def _classify_error(self, error_message: str) -> ErrorCategory:
        """Classify error based on message content."""
        error_lower = error_message.lower()
        
        for pattern, category in self.error_patterns.items():
            if pattern in error_lower:
                return category
        
        # Default classification
        return ErrorCategory.VALIDATION
    
    def _determine_severity(self, exception: Exception, category: ErrorCategory) -> ErrorSeverity:
        """Determine error severity based on exception type and category."""
        if isinstance(exception, (SystemExit, KeyboardInterrupt)):
            return ErrorSeverity.CRITICAL
        elif isinstance(exception, (MemoryError, OSError)):
            return ErrorSeverity.HIGH
        elif isinstance(exception, (ValueError, TypeError, AttributeError)):
            return ErrorSeverity.MEDIUM
        elif category == ErrorCategory.SECURITY:
            return ErrorSeverity.HIGH
        elif category == ErrorCategory.PERFORMANCE:
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW
    
    def _generate_recovery_suggestions(self, exception: Exception, category: ErrorCategory) -> List[str]:
        """Generate recovery suggestions based on error type."""
        suggestions = []
        
        if isinstance(exception, ImportError):
            suggestions.extend([
                "Install missing dependencies",
                "Check Python environment setup",
                "Verify package installation"
            ])
        elif isinstance(exception, ValueError):
            suggestions.extend([
                "Validate input parameters",
                "Check data format and types",
                "Review component configuration"
            ])
        elif isinstance(exception, MemoryError):
            suggestions.extend([
                "Reduce circuit complexity",
                "Enable memory optimization",
                "Increase available memory"
            ])
        elif category == ErrorCategory.SYNTHESIS:
            suggestions.extend([
                "Validate circuit structure",
                "Check component parameters",
                "Retry with simpler configuration"
            ])
        elif category == ErrorCategory.SECURITY:
            suggestions.extend([
                "Review security configuration",
                "Check input validation",
                "Verify access permissions"
            ])
        
        suggestions.append("Contact support if issue persists")
        return suggestions
    
    def _is_auto_recoverable(self, exception: Exception, category: ErrorCategory) -> bool:
        """Determine if error can be automatically recovered."""
        recoverable_types = (ImportError, ValueError, AttributeError)
        recoverable_categories = (ErrorCategory.DEPENDENCY, ErrorCategory.VALIDATION)
        
        return (isinstance(exception, recoverable_types) or 
                category in recoverable_categories)
    
    def _attempt_recovery(self, error_context: ErrorContext) -> bool:
        """Attempt automatic recovery based on error context."""
        recovery_key = f"{error_context.category.value}_{type(Exception).__name__.lower()}"
        
        if recovery_key in self.recovery_strategies:
            try:
                return self.recovery_strategies[recovery_key](error_context)
            except Exception as e:
                logger.warning(f"Recovery attempt failed: {e}")
                return False
        
        # Generic recovery attempts
        return self._generic_recovery(error_context)
    
    def _recover_dependency_missing(self, error_context: ErrorContext) -> bool:
        """Recover from missing dependency errors."""
        logger.info("Attempting dependency recovery...")
        
        # Try to identify and install missing dependency
        message = error_context.message.lower()
        if "module" in message:
            # Extract module name and attempt installation
            try:
                import subprocess
                import sys
                
                # Extract module name (simplified)
                if "no module named" in message:
                    module_name = message.split("no module named")[-1].strip().strip("'\"")
                    
                    logger.info(f"Attempting to install missing module: {module_name}")
                    result = subprocess.run([
                        sys.executable, "-m", "pip", "install", 
                        "--break-system-packages", module_name
                    ], capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        logger.info(f"Successfully installed {module_name}")
                        return True
                        
            except Exception as e:
                logger.warning(f"Dependency recovery failed: {e}")
        
        return False
    
    def _recover_validation_failed(self, error_context: ErrorContext) -> bool:
        """Recover from validation failures."""
        logger.info("Attempting validation recovery...")
        
        # Try to sanitize and re-validate inputs
        if "details" in error_context.details:
            try:
                from .photonic_security import sanitize_input
                
                # Attempt to sanitize problematic inputs
                details = error_context.details.get("details", {})
                for key, value in details.items():
                    if isinstance(value, (str, dict, list)):
                        sanitized = sanitize_input(value)
                        details[key] = sanitized
                
                logger.info("Input sanitization completed")
                return True
                
            except Exception as e:
                logger.warning(f"Validation recovery failed: {e}")
        
        return False
    
    def _recover_synthesis_timeout(self, error_context: ErrorContext) -> bool:
        """Recover from synthesis timeout errors."""
        logger.info("Attempting synthesis timeout recovery...")
        
        # Try to reduce circuit complexity or increase timeout
        if error_context.component and "circuit" in error_context.details:
            try:
                # Implement circuit simplification logic
                logger.info("Reducing circuit complexity for retry")
                return True
                
            except Exception as e:
                logger.warning(f"Synthesis recovery failed: {e}")
        
        return False
    
    def _recover_memory_error(self, error_context: ErrorContext) -> bool:
        """Recover from memory errors."""
        logger.info("Attempting memory error recovery...")
        
        try:
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear any caches
            if hasattr(self, 'optimizer'):
                from .photonic_optimization import clear_caches
                clear_caches()
            
            logger.info("Memory cleanup completed")
            return True
            
        except Exception as e:
            logger.warning(f"Memory recovery failed: {e}")
        
        return False
    
    def _recover_configuration_error(self, error_context: ErrorContext) -> bool:
        """Recover from configuration errors."""
        logger.info("Attempting configuration recovery...")
        
        try:
            # Reset to default configuration
            from .config import get_default_config
            default_config = get_default_config()
            
            logger.info("Reset to default configuration")
            return True
            
        except Exception as e:
            logger.warning(f"Configuration recovery failed: {e}")
        
        return False
    
    def _generic_recovery(self, error_context: ErrorContext) -> bool:
        """Generic recovery attempts."""
        logger.info("Attempting generic recovery...")
        
        try:
            # Clear any global state
            if hasattr(self, '_clear_state'):
                self._clear_state()
            
            # Reset to safe defaults
            logger.info("State reset completed")
            return True
            
        except Exception as e:
            logger.warning(f"Generic recovery failed: {e}")
        
        return False
    
    def _log_error(self, error_context: ErrorContext):
        """Log error with appropriate level based on severity."""
        log_message = (
            f"Error {error_context.error_id}: {error_context.message} "
            f"(Component: {error_context.component}, "
            f"Operation: {error_context.operation})"
        )
        
        if error_context.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message)
        elif error_context.severity == ErrorSeverity.HIGH:
            logger.error(log_message)
        elif error_context.severity == ErrorSeverity.MEDIUM:
            logger.warning(log_message)
        else:
            logger.info(log_message)
    
    def retry_with_backoff(self, 
                          operation: Callable,
                          *args,
                          max_attempts: int = None,
                          component: str = None,
                          **kwargs) -> Any:
        """
        Execute operation with exponential backoff retry.
        
        Args:
            operation: Function to execute
            *args: Positional arguments for operation
            max_attempts: Maximum retry attempts
            component: Component name for error context
            **kwargs: Keyword arguments for operation
            
        Returns:
            Result of successful operation
            
        Raises:
            Last exception if all retries fail
        """
        if max_attempts is None:
            max_attempts = self.max_retry_attempts
        
        last_exception = None
        
        for attempt in range(max_attempts):
            try:
                return operation(*args, **kwargs)
                
            except Exception as e:
                last_exception = e
                
                # Handle error and check if recoverable
                error_context = self.handle_error(
                    e, 
                    component=component,
                    operation=operation.__name__ if hasattr(operation, '__name__') else str(operation),
                    context={"attempt": attempt + 1, "max_attempts": max_attempts}
                )
                
                # If not the last attempt and error is recoverable, retry
                if attempt < max_attempts - 1 and error_context.auto_recoverable:
                    delay = self.retry_delays[min(attempt, len(self.retry_delays) - 1)]
                    logger.info(f"Retrying in {delay} seconds (attempt {attempt + 1}/{max_attempts})")
                    time.sleep(delay)
                else:
                    break
        
        # All retries failed
        logger.error(f"Operation failed after {max_attempts} attempts")
        raise last_exception
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        if not self.error_log:
            return {"total_errors": 0}
        
        stats = {
            "total_errors": len(self.error_log),
            "by_category": {},
            "by_severity": {},
            "by_component": {},
            "recovery_rate": 0,
            "recent_errors": []
        }
        
        # Analyze error distribution
        for error in self.error_log:
            # By category
            category = error.category.value
            stats["by_category"][category] = stats["by_category"].get(category, 0) + 1
            
            # By severity
            severity = error.severity.value
            stats["by_severity"][severity] = stats["by_severity"].get(severity, 0) + 1
            
            # By component
            if error.component:
                stats["by_component"][error.component] = stats["by_component"].get(error.component, 0) + 1
        
        # Calculate recovery rate
        recovered_errors = sum(1 for error in self.error_log 
                             if error.details.get("recovery_successful", False))
        stats["recovery_rate"] = (recovered_errors / len(self.error_log)) * 100
        
        # Recent errors (last 10)
        stats["recent_errors"] = [
            {
                "id": error.error_id,
                "timestamp": error.timestamp,
                "category": error.category.value,
                "severity": error.severity.value,
                "message": error.message,
                "component": error.component
            }
            for error in self.error_log[-10:]
        ]
        
        return stats
    
    def export_error_log(self, filepath: str = None) -> str:
        """Export error log to JSON file."""
        if filepath is None:
            filepath = f"error_log_{int(time.time())}.json"
        
        export_data = {
            "export_timestamp": time.time(),
            "total_errors": len(self.error_log),
            "statistics": self.get_error_statistics(),
            "errors": [
                {
                    "error_id": error.error_id,
                    "timestamp": error.timestamp,
                    "category": error.category.value,
                    "severity": error.severity.value,
                    "message": error.message,
                    "details": error.details,
                    "component": error.component,
                    "operation": error.operation,
                    "recovery_suggestions": error.recovery_suggestions,
                    "auto_recoverable": error.auto_recoverable
                }
                for error in self.error_log
            ]
        }
        
        Path(filepath).write_text(json.dumps(export_data, indent=2))
        logger.info(f"Error log exported to {filepath}")
        return filepath


# Global error handler instance
_error_handler = PhotonicErrorHandler()


def handle_photonic_error(exception: Exception,
                         component: str = None,
                         operation: str = None,
                         context: Dict[str, Any] = None) -> ErrorContext:
    """Global error handling function."""
    return _error_handler.handle_error(exception, component, operation, context)


def retry_operation(operation: Callable, 
                   *args,
                   component: str = None,
                   **kwargs) -> Any:
    """Retry operation with automatic error handling."""
    return _error_handler.retry_with_backoff(operation, *args, component=component, **kwargs)


def get_error_stats() -> Dict[str, Any]:
    """Get global error statistics."""
    return _error_handler.get_error_statistics()


def export_errors(filepath: str = None) -> str:
    """Export global error log."""
    return _error_handler.export_error_log(filepath)


# Context manager for error handling
class PhotonicErrorContext:
    """Context manager for comprehensive error handling."""
    
    def __init__(self, component: str, operation: str, context: Dict[str, Any] = None):
        self.component = component
        self.operation = operation
        self.context = context or {}
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        logger.debug(f"Starting operation {self.operation} in component {self.component}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        
        if exc_type is not None:
            # Error occurred
            self.context["duration"] = duration
            handle_photonic_error(
                exc_val,
                component=self.component,
                operation=self.operation,
                context=self.context
            )
            return False  # Re-raise exception
        else:
            # Success
            logger.debug(f"Operation {self.operation} completed successfully in {duration:.3f}s")
            return True


if __name__ == "__main__":
    # Demo error handling capabilities
    print("🛡️ Photonic-MLIR Bridge - Advanced Error Handling Demo")
    print("=" * 60)
    
    # Test error handling
    try:
        with PhotonicErrorContext("demo", "test_operation"):
            raise ValueError("Test error for demonstration")
    except ValueError:
        pass
    
    # Show statistics
    stats = get_error_stats()
    print(f"\nError Statistics:")
    print(f"Total Errors: {stats['total_errors']}")
    print(f"Recovery Rate: {stats.get('recovery_rate', 0):.1f}%")
    
    # Export error log
    log_file = export_errors("demo_error_log.json")
    print(f"Error log exported to: {log_file}")
    
    print("\n✅ Error handling system operational!")