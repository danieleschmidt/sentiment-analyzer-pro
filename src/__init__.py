from importlib import metadata

# Safe imports for sentiment analysis components (may have missing dependencies)
try:
    from .aspect_sentiment import extract_aspects, predict_aspect_sentiment
    from .model_comparison import compare_models
    from .models import (
        build_lstm_model,
        build_model,
        build_nb_model,
        build_transformer_model,
    )
    from .evaluate import analyze_errors, compute_confusion, evaluate, cross_validate
    from .preprocessing import clean_text, remove_stopwords, lemmatize_tokens
    _sentiment_available = True
except ImportError:
    # Graceful fallback for missing dependencies
    extract_aspects = None
    predict_aspect_sentiment = None
    compare_models = None
    build_lstm_model = None
    build_model = None
    build_nb_model = None
    build_transformer_model = None
    analyze_errors = None
    compute_confusion = None
    evaluate = None
    cross_validate = None
    clean_text = None
    remove_stopwords = None
    lemmatize_tokens = None
    _sentiment_available = False

# Always import photonic components (should work without external dependencies)
from .photonic_init import get_photonic_status, check_autonomous_readiness

try:  # optional dependency: argparse + CLI utilities
    from .cli import main as cli_main
except Exception:  # pragma: no cover - optional CLI
    cli_main = None  # type: ignore

try:  # optional dependency: flask web server
    from .webapp import app as web_app, main as web_main
except Exception:  # pragma: no cover - optional web server
    web_app = None  # type: ignore
    web_main = None  # type: ignore

try:
    __version__ = metadata.version("sentiment-analyzer-pro")
except metadata.PackageNotFoundError:  # pragma: no cover - package not installed
    __version__ = "0.0.0"

__all__ = [
    # Sentiment analysis components (may be None if dependencies missing)
    "build_model",
    "build_nb_model", 
    "build_lstm_model",
    "build_transformer_model",
    "compare_models",
    "clean_text",
    "remove_stopwords",
    "lemmatize_tokens",
    "extract_aspects", 
    "predict_aspect_sentiment",
    "evaluate",
    "compute_confusion",
    "analyze_errors",
    "cross_validate",
    # Optional components
    "cli_main",
    "web_app",
    "web_main",
    # Photonic-MLIR bridge components (always available)
    "get_photonic_status",
    "check_autonomous_readiness",
    # Package info
    "__version__",
    "_sentiment_available",
]
