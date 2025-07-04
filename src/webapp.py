"""Flask web server for sentiment predictions."""
from __future__ import annotations

import argparse
import logging
import os
from functools import lru_cache
from typing import Any

from flask import Flask, jsonify, request
from importlib import metadata
from pydantic import ValidationError

import joblib

from .schemas import PredictRequest

from .models import SentimentModel

app = Flask(__name__)
logger = logging.getLogger(__name__)

REQUEST_COUNT = 0
PREDICTION_COUNT = 0

MODEL_PATH = os.getenv("MODEL_PATH", "model.joblib")

try:
    APP_VERSION = metadata.version("sentiment-analyzer-pro")
except metadata.PackageNotFoundError:  # pragma: no cover - local usage
    APP_VERSION = "0.0.0"


@lru_cache(maxsize=1)
def load_model(path: str = MODEL_PATH) -> SentimentModel:
    """Load and cache a trained model from disk."""
    return joblib.load(path)


@app.before_request
def _log_request() -> None:  # pragma: no cover - logging side effect
    global REQUEST_COUNT
    REQUEST_COUNT += 1
    logger.info("%s %s", request.method, request.path)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    try:
        req = PredictRequest(**data)
    except ValidationError as exc:
        return jsonify({"error": "Invalid input", "details": exc.errors()}), 400
    model = load_model(MODEL_PATH)
    prediction = model.predict([req.text])[0]
    global PREDICTION_COUNT
    PREDICTION_COUNT += 1
    return jsonify({"prediction": prediction})


@app.route("/", methods=["GET"])
def index():
    """Simple health check endpoint."""
    return jsonify({"status": "ok"})


@app.route("/version", methods=["GET"])
def version():
    """Return the running package version."""
    return jsonify({"version": APP_VERSION})


@app.route("/metrics", methods=["GET"])
def metrics() -> Any:
    """Return simple service metrics."""
    return jsonify({"requests": REQUEST_COUNT, "predictions": PREDICTION_COUNT})


def main(argv: list[str] | None = None) -> None:
    global MODEL_PATH
    parser = argparse.ArgumentParser(description="Run prediction web server")
    parser.add_argument("--model", default=MODEL_PATH, help="Trained model path")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=5000, type=int)
    args = parser.parse_args(argv)

    MODEL_PATH = args.model
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":  # pragma: no cover - manual launch
    main()
