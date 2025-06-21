from .aspect_sentiment import extract_aspects, predict_aspect_sentiment
from .model_comparison import compare_models
from .models import build_lstm_model, build_model, build_transformer_model
from .evaluate import analyze_errors, compute_confusion, evaluate
from .preprocessing import clean_text, remove_stopwords

try:  # optional dependency: argparse + CLI utilities
    from .cli import main as cli_main
except Exception:  # pragma: no cover - optional CLI
    cli_main = None  # type: ignore

try:  # optional dependency: flask web server
    from .webapp import app as web_app, main as web_main
except Exception:  # pragma: no cover - optional web server
    web_app = None  # type: ignore
    web_main = None  # type: ignore

__all__ = [
    "build_model",
    "build_lstm_model",
    "build_transformer_model",
    "compare_models",
    "clean_text",
    "remove_stopwords",
    "extract_aspects",
    "predict_aspect_sentiment",
    "evaluate",
    "compute_confusion",
    "analyze_errors",
    "cli_main",
    "web_app",
    "web_main",
]
