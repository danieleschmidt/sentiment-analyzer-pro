"""Flask web server for sentiment predictions."""
from __future__ import annotations

import argparse
from typing import Dict

from flask import Flask, jsonify, request
import joblib

from .models import SentimentModel

app = Flask(__name__)

MODEL_PATH = "model.joblib"
_model_cache: Dict[str, SentimentModel] = {}


def load_model(path: str = MODEL_PATH) -> SentimentModel:
    """Load and cache a trained model from disk."""
    if path not in _model_cache:
        _model_cache[path] = joblib.load(path)
    return _model_cache[path]


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    text = data.get("text")
    if not text:
        return jsonify({"error": "Missing text"}), 400
    model = load_model(MODEL_PATH)
    prediction = model.predict([text])[0]
    return jsonify({"prediction": prediction})


def main(argv: list[str] | None = None) -> None:
    global MODEL_PATH
    parser = argparse.ArgumentParser(description="Run prediction web server")
    parser.add_argument("--model", default=MODEL_PATH, help="Trained model path")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=5000, type=int)
    args = parser.parse_args(argv)

    MODEL_PATH = args.model
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":  # pragma: no cover - manual launch
    main()
