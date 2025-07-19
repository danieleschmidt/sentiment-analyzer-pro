from pydantic import BaseModel, Field, field_validator
from typing import Iterable
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


def validate_columns(columns: Iterable[str], required: Iterable[str]) -> None:
    missing = [c for c in required if c not in columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
