"""
sentiment-analyzer-pro: Multi-level sentiment analysis toolkit.

Provides lexicon-based scoring, aspect extraction, and emotion classification
without requiring any ML frameworks — just Python stdlib + numpy (optional).
"""

from .lexicon import LexiconSentiment
from .aspects import AspectExtractor
from .emotions import EmotionClassifier
from .analyzer import SentimentAnalyzerPro, SentimentResult

__all__ = [
    "LexiconSentiment",
    "AspectExtractor",
    "EmotionClassifier",
    "SentimentAnalyzerPro",
    "SentimentResult",
]

__version__ = "1.0.0"
