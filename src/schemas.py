from pydantic import BaseModel, Field, field_validator
from typing import Iterable, List
import re


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, description="Text to analyze")
    
    @field_validator('text')
    @classmethod
    def sanitize_text(cls, v):
        """Sanitize input text to prevent injection attacks."""
        if not isinstance(v, str):
            raise ValueError('Text must be a string')
        
        # Remove any potential script tags or suspicious patterns
        v = re.sub(r'<script[^>]*>.*?</script>', '', v, flags=re.IGNORECASE | re.DOTALL)
        v = re.sub(r'javascript:', '', v, flags=re.IGNORECASE)
        v = re.sub(r'on\w+\s*=', '', v, flags=re.IGNORECASE)
        
        # Strip excessive whitespace
        v = re.sub(r'\s+', ' ', v.strip())
        
        return v


class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=100, description="List of texts to analyze")
    
    @field_validator('texts')
    @classmethod
    def sanitize_texts(cls, v):
        """Sanitize input texts to prevent injection attacks."""
        if not isinstance(v, list):
            raise ValueError('Texts must be a list')
        
        sanitized = []
        for text in v:
            if not isinstance(text, str):
                raise ValueError('Each text must be a string')
            
            if len(text) > 10000:
                raise ValueError('Individual text length cannot exceed 10000 characters')
            
            # Apply same sanitization as single text
            text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
            text = re.sub(r'javascript:', '', text, flags=re.IGNORECASE)
            text = re.sub(r'on\w+\s*=', '', text, flags=re.IGNORECASE)
            text = re.sub(r'\s+', ' ', text.strip())
            
            if len(text) < 1:
                raise ValueError('Text cannot be empty after sanitization')
            
            sanitized.append(text)
        
        return sanitized


def validate_columns(columns: Iterable[str], required: Iterable[str]) -> None:
    missing = [c for c in required if c not in columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
