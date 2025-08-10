"""
Core API functionality for sentiment analyzer
Generation 1: Make It Work - Essential API endpoints
"""
import time
import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
from flask import Flask, request, jsonify, g
from functools import wraps
import traceback

from .enhanced_config import get_config
from .health_check import quick_health_check
from .models import build_nb_model
from .preprocessing import preprocess_text

logger = logging.getLogger(__name__)

class SentimentLabel(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

@dataclass
class PredictionRequest:
    text: str
    include_confidence: bool = False
    include_preprocessing: bool = False

@dataclass
class PredictionResponse:
    text: str
    sentiment: str
    confidence: Optional[float] = None
    processed_text: Optional[str] = None
    processing_time_ms: float = 0
    model_version: str = "nb_v1"

class SentimentAPI:
    """Core sentiment analysis API"""
    
    def __init__(self, model=None, config=None):
        self.config = config or get_config()
        self.model = model
        self._model_loaded = False
        self.request_count = 0
        self.prediction_count = 0
        self._load_model()
    
    def _load_model(self):
        """Load the sentiment analysis model"""
        try:
            if self.model is None:
                logger.info("Loading default Naive Bayes model...")
                self.model = build_nb_model()
            self._model_loaded = True
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self._model_loaded = False
    
    def predict_sentiment(self, 
                         text: str, 
                         include_confidence: bool = False,
                         include_preprocessing: bool = False) -> PredictionResponse:
        """
        Predict sentiment for given text
        
        Args:
            text: Input text to analyze
            include_confidence: Whether to include confidence scores
            include_preprocessing: Whether to include preprocessed text
        
        Returns:
            PredictionResponse with sentiment and optional metadata
        """
        start_time = time.time()
        
        try:
            if not self._model_loaded:
                raise ValueError("Model not loaded")
            
            # Preprocess text
            processed_text = preprocess_text(text)
            
            # Simple sentiment prediction logic (placeholder)
            # In a real implementation, this would use the trained model
            sentiment = self._simple_sentiment_prediction(processed_text)
            confidence = None
            
            if include_confidence:
                confidence = self._calculate_confidence(processed_text, sentiment)
            
            processing_time = (time.time() - start_time) * 1000
            
            response = PredictionResponse(
                text=text,
                sentiment=sentiment,
                confidence=confidence if include_confidence else None,
                processed_text=processed_text if include_preprocessing else None,
                processing_time_ms=processing_time,
                model_version="nb_v1"
            )
            
            self.prediction_count += 1
            logger.debug(f"Prediction completed in {processing_time:.2f}ms")
            
            return response
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            processing_time = (time.time() - start_time) * 1000
            
            return PredictionResponse(
                text=text,
                sentiment="error",
                confidence=None,
                processed_text=None,
                processing_time_ms=processing_time,
                model_version="error"
            )
    
    def _simple_sentiment_prediction(self, text: str) -> str:
        """Simple rule-based sentiment prediction (placeholder)"""
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 
                         'fantastic', 'love', 'like', 'best', 'awesome']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'worst', 
                         'horrible', 'disgusting', 'disappointing', 'poor']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return SentimentLabel.POSITIVE.value
        elif negative_count > positive_count:
            return SentimentLabel.NEGATIVE.value
        else:
            return SentimentLabel.NEUTRAL.value
    
    def _calculate_confidence(self, text: str, sentiment: str) -> float:
        """Calculate confidence score (placeholder)"""
        # Simple confidence calculation based on text length and sentiment strength
        text_length_factor = min(len(text.split()) / 10, 1.0)
        base_confidence = 0.6 + (text_length_factor * 0.3)
        
        # Add some randomness to simulate model uncertainty
        import random
        noise = random.uniform(-0.1, 0.1)
        confidence = max(0.1, min(0.99, base_confidence + noise))
        
        return round(confidence, 3)
    
    def predict_batch(self, texts: List[str], **kwargs) -> List[PredictionResponse]:
        """Predict sentiment for multiple texts"""
        return [self.predict_sentiment(text, **kwargs) for text in texts]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get API usage statistics"""
        return {
            "model_loaded": self._model_loaded,
            "model_version": "nb_v1",
            "total_requests": self.request_count,
            "total_predictions": self.prediction_count,
            "uptime_seconds": time.time() - getattr(self, '_start_time', time.time())
        }

def create_app(config=None) -> Flask:
    """Create Flask application with sentiment analysis API"""
    app = Flask(__name__)
    app.config.update(
        JSON_SORT_KEYS=False,
        JSONIFY_PRETTYPRINT_REGULAR=True
    )
    
    # Initialize API
    sentiment_api = SentimentAPI(config=config)
    sentiment_api._start_time = time.time()
    
    def track_requests(f):
        """Decorator to track API requests"""
        @wraps(f)
        def decorated_function(*args, **kwargs):
            sentiment_api.request_count += 1
            g.start_time = time.time()
            return f(*args, **kwargs)
        return decorated_function
    
    @app.route('/health', methods=['GET'])
    @track_requests
    def health_check():
        """Health check endpoint"""
        try:
            health_data = quick_health_check()
            return jsonify(health_data), 200
        except Exception as e:
            return jsonify({
                "error": "Health check failed",
                "message": str(e)
            }), 500
    
    @app.route('/predict', methods=['POST'])
    @track_requests
    def predict():
        """Predict sentiment for text"""
        try:
            data = request.get_json()
            
            if not data or 'text' not in data:
                return jsonify({
                    "error": "Missing required field 'text'"
                }), 400
            
            text = data['text']
            include_confidence = data.get('include_confidence', False)
            include_preprocessing = data.get('include_preprocessing', False)
            
            if not isinstance(text, str) or not text.strip():
                return jsonify({
                    "error": "Text must be a non-empty string"
                }), 400
            
            response = sentiment_api.predict_sentiment(
                text=text,
                include_confidence=include_confidence,
                include_preprocessing=include_preprocessing
            )
            
            return jsonify({
                "text": response.text,
                "sentiment": response.sentiment,
                "confidence": response.confidence,
                "processed_text": response.processed_text,
                "processing_time_ms": response.processing_time_ms,
                "model_version": response.model_version
            }), 200
            
        except Exception as e:
            logger.error(f"Prediction endpoint error: {e}")
            return jsonify({
                "error": "Prediction failed",
                "message": str(e),
                "traceback": traceback.format_exc() if app.debug else None
            }), 500
    
    @app.route('/predict/batch', methods=['POST'])
    @track_requests
    def predict_batch():
        """Predict sentiment for multiple texts"""
        try:
            data = request.get_json()
            
            if not data or 'texts' not in data:
                return jsonify({
                    "error": "Missing required field 'texts'"
                }), 400
            
            texts = data['texts']
            if not isinstance(texts, list) or not texts:
                return jsonify({
                    "error": "Texts must be a non-empty list"
                }), 400
            
            if len(texts) > 100:
                return jsonify({
                    "error": "Maximum 100 texts allowed per request"
                }), 400
            
            include_confidence = data.get('include_confidence', False)
            include_preprocessing = data.get('include_preprocessing', False)
            
            responses = sentiment_api.predict_batch(
                texts,
                include_confidence=include_confidence,
                include_preprocessing=include_preprocessing
            )
            
            return jsonify({
                "results": [{
                    "text": r.text,
                    "sentiment": r.sentiment,
                    "confidence": r.confidence,
                    "processed_text": r.processed_text,
                    "processing_time_ms": r.processing_time_ms,
                    "model_version": r.model_version
                } for r in responses],
                "total_processed": len(responses)
            }), 200
            
        except Exception as e:
            logger.error(f"Batch prediction error: {e}")
            return jsonify({
                "error": "Batch prediction failed",
                "message": str(e)
            }), 500
    
    @app.route('/stats', methods=['GET'])
    @track_requests
    def stats():
        """Get API statistics"""
        try:
            stats_data = sentiment_api.get_stats()
            return jsonify(stats_data), 200
        except Exception as e:
            return jsonify({
                "error": "Failed to get stats",
                "message": str(e)
            }), 500
    
    @app.route('/', methods=['GET'])
    @track_requests
    def root():
        """Root endpoint"""
        return jsonify({
            "service": "Sentiment Analyzer Pro",
            "version": "1.0.0",
            "status": "operational",
            "endpoints": {
                "health": "/health",
                "predict": "/predict",
                "batch_predict": "/predict/batch",
                "stats": "/stats"
            }
        }), 200
    
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            "error": "Not found",
            "message": "Endpoint not found"
        }), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({
            "error": "Internal server error",
            "message": "An unexpected error occurred"
        }), 500
    
    return app

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app = create_app()
    config = get_config()
    app.run(
        host=config.server.host,
        port=config.server.port,
        debug=config.server.debug
    )