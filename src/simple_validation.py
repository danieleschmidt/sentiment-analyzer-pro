
from typing import Any, Dict, List, Optional
import re

class SimpleValidator:
    """Basic input validation for Generation 1."""
    
    @staticmethod
    def validate_text(text: Any) -> str:
        """Validate and clean text input."""
        if not text:
            raise ValueError("Text input cannot be empty")
        
        if not isinstance(text, str):
            text = str(text)
        
        # Basic sanitization
        text = text.strip()
        if len(text) > 10000:  # Reasonable limit
            raise ValueError("Text too long (max 10000 characters)")
        
        return text
    
    @staticmethod
    def validate_model_params(params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate model parameters."""
        validated = {}
        
        # Common parameter validation
        if 'batch_size' in params:
            batch_size = params['batch_size']
            if not isinstance(batch_size, int) or batch_size < 1 or batch_size > 1000:
                raise ValueError("batch_size must be integer between 1-1000")
            validated['batch_size'] = batch_size
        
        if 'learning_rate' in params:
            lr = params['learning_rate']
            if not isinstance(lr, (int, float)) or lr <= 0 or lr > 1:
                raise ValueError("learning_rate must be float between 0-1")
            validated['learning_rate'] = float(lr)
        
        return validated

# Global validator instance
validator = SimpleValidator()
