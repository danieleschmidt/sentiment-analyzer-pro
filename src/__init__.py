from importlib import metadata

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
    "cli_main",
    "web_app",
    "web_main",
    "__version__",
]
