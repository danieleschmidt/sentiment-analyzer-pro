
import logging
import traceback
from functools import wraps
from typing import Callable, Any

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def handle_errors(func: Callable) -> Callable:
    """Simple error handling decorator."""
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    return wrapper

class SimpleErrorHandler:
    """Basic error handling and recovery system."""
    
    def __init__(self):
        self.error_count = 0
        self.errors = []
    
    def record_error(self, error: Exception, context: str = ""):
        """Record an error for tracking."""
        self.error_count += 1
        self.errors.append({
            "error": str(error),
            "context": context,
            "timestamp": time.time(),
            "type": type(error).__name__
        })
        
        # Keep only last 100 errors
        if len(self.errors) > 100:
            self.errors = self.errors[-100:]
    
    def get_error_stats(self):
        """Get error statistics."""
        return {
            "total_errors": self.error_count,
            "recent_errors": len(self.errors),
            "error_types": list(set([e["type"] for e in self.errors]))
        }

# Global error handler
error_handler = SimpleErrorHandler()
