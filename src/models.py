"""Baseline sentiment classifier using Logistic Regression."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import threading
from typing import Dict, Any, Optional
import time

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline
except Exception:  # pragma: no cover - optional dependency
    TfidfVectorizer = LogisticRegression = MultinomialNB = Pipeline = None

try:
    from tensorflow import keras
except Exception:  # pragma: no cover - optional dependency
    keras = None

try:
    from transformers import DistilBertConfig, DistilBertForSequenceClassification
except Exception:  # pragma: no cover - optional dependency
    DistilBertConfig = None
    DistilBertForSequenceClassification = None

try:
    from .neuromorphic_spikeformer import NeuromorphicSentimentAnalyzer, SpikeformerConfig
except Exception:  # pragma: no cover - optional dependency
    NeuromorphicSentimentAnalyzer = None
    SpikeformerConfig = None


class ModelCache:
    """Thread-safe model cache with performance monitoring."""
    
    def __init__(self, max_size: int = 128, ttl_seconds: int = 3600):
        self._cache: Dict[str, Any] = {}
        self._timestamps: Dict[str, float] = {}
        self._lock = threading.RLock()
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._hits = 0
        self._misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            current_time = time.time()
            
            if key in self._cache:
                # Check TTL
                if current_time - self._timestamps[key] < self._ttl:
                    self._hits += 1
                    return self._cache[key]
                else:
                    # Expired
                    del self._cache[key]
                    del self._timestamps[key]
            
            self._misses += 1
            return None
    
    def put(self, key: str, value: Any) -> None:
        with self._lock:
            current_time = time.time()
            
            # Evict oldest entries if cache is full
            if len(self._cache) >= self._max_size:
                oldest_key = min(self._timestamps.keys(), key=lambda k: self._timestamps[k])
                del self._cache[oldest_key]
                del self._timestamps[oldest_key]
            
            self._cache[key] = value
            self._timestamps[key] = current_time
    
    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0
            return {
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': hit_rate,
                'cache_size': len(self._cache)
            }


# Global caches
_prediction_cache = ModelCache(max_size=1000, ttl_seconds=300)  # 5 min TTL for predictions
_model_cache = ModelCache(max_size=10, ttl_seconds=3600)  # 1 hour TTL for models


@dataclass
class SentimentModel:
    pipeline: Pipeline
    _prediction_cache: Optional[ModelCache] = None

    def __post_init__(self):
        if self._prediction_cache is None:
            self._prediction_cache = _prediction_cache

    def fit(self, texts, labels):
        self.pipeline.fit(texts, labels)
        # Clear cache when model is retrained
        if self._prediction_cache:
            self._prediction_cache._cache.clear()

    @lru_cache(maxsize=1000)
    def _predict_single_cached(self, text: str) -> str:
        """Cache single predictions using LRU cache."""
        return self.pipeline.predict([text])[0]

    def predict(self, texts):
        """Optimized prediction with caching for repeated inputs."""
        if isinstance(texts, str):
            texts = [texts]
        
        results = []
        for text in texts:
            # For single text prediction, use caching
            if len(texts) == 1:
                cache_key = f"pred_{hash(text)}"
                cached_result = self._prediction_cache.get(cache_key) if self._prediction_cache else None
                
                if cached_result is not None:
                    results.append(cached_result)
                else:
                    prediction = self.pipeline.predict([text])[0]
                    if self._prediction_cache:
                        self._prediction_cache.put(cache_key, prediction)
                    results.append(prediction)
            else:
                # For batch predictions, skip caching to avoid overhead
                results.append(self.pipeline.predict([text])[0])
        
        # Return single result for single input, list for multiple inputs
        if len(texts) == 1:
            return results[0]
        else:
            return results
    
    def predict_proba(self, texts):
        """Get prediction probabilities if available."""
        if hasattr(self.pipeline, 'predict_proba'):
            return self.pipeline.predict_proba(texts)
        return None
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get caching performance statistics."""
        return self._prediction_cache.get_stats() if self._prediction_cache else {}


def build_model() -> SentimentModel:
    if Pipeline is None or TfidfVectorizer is None or LogisticRegression is None:
        raise ImportError("scikit-learn is required for build_model")
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
    return SentimentModel(pipeline=pipeline)


def build_nb_model() -> SentimentModel:
    """Return a simple Naive Bayes sentiment classifier."""
    if (
        Pipeline is None
        or TfidfVectorizer is None
        or MultinomialNB is None
    ):
        raise ImportError("scikit-learn is required for build_nb_model")
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("clf", MultinomialNB()),
        ]
    )
    return SentimentModel(pipeline=pipeline)


def build_lstm_model(
    vocab_size: int = 10000, embed_dim: int = 128, sequence_length: int = 100
) -> keras.Model:
    """Return a simple LSTM-based sentiment classifier."""
    if keras is None:
        raise ImportError("TensorFlow is required for build_lstm_model")
    model = keras.Sequential(
        [
            keras.layers.Embedding(vocab_size, embed_dim, input_length=sequence_length),
            keras.layers.LSTM(64),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def build_transformer_model(num_labels: int = 2) -> DistilBertForSequenceClassification:
    """Return a minimal DistilBERT model for sentiment classification."""
    if DistilBertForSequenceClassification is None or DistilBertConfig is None:
        raise ImportError("transformers is required for build_transformer_model")
    config = DistilBertConfig(vocab_size=30522, num_labels=num_labels)
    model = DistilBertForSequenceClassification(config)
    return model


def build_neuromorphic_model(config: dict = None) -> NeuromorphicSentimentAnalyzer:
    """
    Return a neuromorphic spikeformer model for bio-inspired sentiment analysis.
    
    Args:
        config: Optional configuration dictionary for spikeformer parameters
        
    Returns:
        NeuromorphicSentimentAnalyzer instance
        
    Raises:
        ImportError: If neuromorphic dependencies are not available
    """
    if NeuromorphicSentimentAnalyzer is None:
        raise ImportError("Neuromorphic spikeformer dependencies are required")
    
    # Configure neuromorphic model
    if config:
        spikeformer_config = SpikeformerConfig(**config)
        analyzer = NeuromorphicSentimentAnalyzer(spikeformer_config)
    else:
        analyzer = NeuromorphicSentimentAnalyzer()
    
    return analyzer


def get_available_models() -> list[str]:
    """
    Return list of available model types based on installed dependencies.
    
    Returns:
        List of available model names
    """
    available = []
    
    if Pipeline is not None:
        available.extend(["logistic_regression", "naive_bayes"])
    
    if keras is not None:
        available.append("lstm")
    
    if DistilBertForSequenceClassification is not None:
        available.append("transformer")
    
    if NeuromorphicSentimentAnalyzer is not None:
        available.append("neuromorphic")
    
    return available
