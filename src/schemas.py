```python
from pydantic import BaseModel, Field, field_validator
from typing import Iterable, List, Optional, Dict, Any
import re
import logging
from pathlib import Path


logger = logging.getLogger(__name__)


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, description="Text to analyze")
    confidence_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum confidence threshold")
    return_probabilities: bool = Field(False, description="Return prediction probabilities")
    
    @field_validator('text')
    @classmethod
    def sanitize_text(cls, v):
        """Sanitize input text to prevent injection attacks."""
        if not isinstance(v, str):
            raise ValueError('Text must be a string')
        
        # Log potentially malicious content for security monitoring
        original_length = len(v)
        
        # Remove any potential script tags or suspicious patterns
        v = re.sub(r'<script[^>]*>.*?</script>', '', v, flags=re.IGNORECASE | re.DOTALL)
        v = re.sub(r'javascript:', '', v, flags=re.IGNORECASE)
        v = re.sub(r'on\w+\s*=', '', v, flags=re.IGNORECASE)
        
        # Check for common SQL injection patterns
        if re.search(r'\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION)\b', v, re.IGNORECASE):
            logger.warning(f"Potentially malicious SQL-like content detected: {v[:100]}...")
        
        # Strip excessive whitespace
        v = re.sub(r'\s+', ' ', v.strip())
        
        if len(v) != original_length:
            logger.info(f"Input text sanitized: {original_length} -> {len(v)} characters")
        
        return v


class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=1000, description="List of texts to analyze")
    confidence_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    return_probabilities: bool = Field(False, description="Return prediction probabilities")
    
    @field_validator('texts')
    @classmethod
    def validate_texts(cls, v):
        """Validate and sanitize batch texts."""
        if not isinstance(v, list):
            raise ValueError('Texts must be a list')
        
        sanitized = []
        for i, text in enumerate(v):
            if not isinstance(text, str):
                raise ValueError(f'Text at index {i} must be a string')
            if len(text.strip()) == 0:
                raise ValueError(f'Text at index {i} cannot be empty')
            if len(text) > 10000:
                raise ValueError(f'Text at index {i} exceeds maximum length of 10000 characters')
            
            # Apply same sanitization as single request
            sanitized_text = PredictRequest.sanitize_text(text)
            sanitized.append(sanitized_text)
        
        return sanitized


class ModelMetadata(BaseModel):
    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    accuracy: Optional[float] = Field(None, ge=0.0, le=1.0, description="Model accuracy")
    training_date: Optional[str] = Field(None, description="Training date")
    features: Optional[List[str]] = Field(None, description="Feature names")


class PredictionResponse(BaseModel):
    prediction: str = Field(..., description="Predicted sentiment")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Prediction confidence")
    probabilities: Optional[Dict[str, float]] = Field(None, description="Class probabilities")
    processing_time_ms: Optional[float] = Field(None, description="Processing time in milliseconds")


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    total_processing_time_ms: Optional[float] = Field(None, description="Total processing time")
    metadata: Optional[ModelMetadata] = Field(None, description="Model metadata")


class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
    error_code: str = Field(..., description="Error code")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")


def validate_columns(columns: Iterable[str], required: Iterable[str]) -> None:
    """Validate that required columns are present."""
    missing = [c for c in required if c not in columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def validate_file_path(file_path: str, allowed_extensions: List[str] = None) -> Path:
    """Validate file path for security and existence."""
    if not isinstance(file_path, (str, Path)):
        raise ValueError("File path must be a string or Path object")
    
    path = Path(file_path)
    
    # Security: Prevent path traversal attacks
    if '..' in str(path) or str(path).startswith('/'):
        if not str(path).startswith(('/tmp', '/var/tmp')):
            raise ValueError("Invalid file path: potential path traversal detected")
    
    # Check file extension
    if allowed_extensions and path.suffix.lower() not in allowed_extensions:
        raise ValueError(f"File extension {path.suffix} not allowed. Allowed: {allowed_extensions}")
    
    return path


def validate_csv_structure(df, required_columns: List[str] = None, max_rows: int = 1000000) -> None:
    """Validate CSV DataFrame structure and size."""
    if df is None or len(df) == 0:
        raise ValueError("CSV file is empty or could not be read")
    
    if len(df) > max_rows:
        raise ValueError(f"CSV file too large: {len(df)} rows (max: {max_rows})")
    
    if required_columns:
        validate_columns(df.columns, required_columns)
    
    # Check for suspicious column names
    suspicious_columns = [col for col in df.columns if 
                         any(pattern in str(col).lower() for pattern in ['password', 'secret', 'key', 'token'])]
    if suspicious_columns:
        logger.warning(f"Potentially sensitive column names detected: {suspicious_columns}")


class ValidationError(Exception):
    """Custom validation error with additional context."""
    
    def __init__(self, message: str, error_code: str = "VALIDATION_ERROR", details: Dict[str, Any] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}


class SecurityError(Exception):
    """Security-related validation error."""
    
    def __init__(self, message: str, error_code: str = "SECURITY_ERROR", details: Dict[str, Any] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
```
