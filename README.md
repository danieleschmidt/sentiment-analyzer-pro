# Sentiment Analyzer Pro

This project aims to develop an advanced sentiment analysis tool. It will start with a basic sentiment classifier and progressively add features like aspect-based sentiment analysis, sophisticated text preprocessing, model comparison, and robust evaluation.

## Project Goals
- Implement and compare various sentiment analysis models (e.g., Naive Bayes, Logistic Regression, RNN/LSTM, Transformer-based).
- Develop robust text preprocessing pipelines.
- Explore aspect-based sentiment analysis.
- Implement comprehensive model evaluation and error analysis.
- Ensure code is well-documented and tested.

## Tech Stack (Planned)
- Python
- Scikit-learn
- NLTK / spaCy
- TensorFlow / PyTorch
- Pandas, NumPy

## Initial File Structure
sentiment-analyzer-pro/
├── data/ # For datasets (e.g., sample CSVs)
│   └── sample_reviews.csv
├── notebooks/ # For exploration and experimentation
│   └── initial_exploration.ipynb
├── src/ # Source code
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── models.py
│   ├── train.py
│   ├── predict.py
│   └── evaluate.py
├── tests/ # Unit tests
│   ├── __init__.py
│   └── test_preprocessing.py
├── requirements.txt
├── .gitignore
└── README.md

## How to Contribute (and test Jules)
New features, bug fixes, and improvements will be requested via GitHub Issues and assigned to Jules, our Async Development Agent.

For details on preparing datasets, see [docs/DATA_HANDLING.md](docs/DATA_HANDLING.md).
For a brief overview of the new aspect-based sentiment component, see [docs/ASPECT_SENTIMENT.md](docs/ASPECT_SENTIMENT.md).
For running detailed evaluations, see [docs/EVALUATION.md](docs/EVALUATION.md).

## Getting Started

1. Install dependencies:
   ```bash
   pip install -e .
   ```
   For advanced models or the web API, install optional extras:
   ```bash
   pip install -e .[ml,web]
   ```
2. Run unit tests:
   ```bash
   pytest -q
   ```
3. Train the baseline model:
   ```bash
   python -m src.train
   ```
4. (Optional) Build the LSTM model for experimentation:
   ```python
   from src.models import build_lstm_model
   model = build_lstm_model()
   ```
5. (Optional) Build a Transformer-based model:
   ```python
   from src.models import build_transformer_model
   model = build_transformer_model()
   ```
6. Make predictions on a CSV file with a `text` column:
   ```bash
   python -m src.predict your_reviews.csv
   ```
7. Preprocess a CSV of raw reviews:
   ```bash
   sentiment-cli preprocess data/raw.csv --out clean.csv --remove-stopwords
   ```
8. Split a labeled dataset into train and test files:
   ```bash
   sentiment-cli split data/all.csv --train train.csv --test test.csv --ratio 0.2
   ```
9. Summarize a dataset with basic statistics:
   ```bash
sentiment-cli summary train.csv
# show the five most common words
sentiment-cli summary train.csv --top 5
```
The `--top` flag lists the most frequent tokens in the dataset. Punctuation is
automatically stripped when counting words, so `movie!` and `movie.` are
aggregated under the same entry.
10. Use the unified CLI for training, prediction, evaluation, and analysis:
   ```bash
sentiment-cli train --csv data/sample_reviews.csv --model my_model.joblib
sentiment-cli predict your_reviews.csv --model my_model.joblib
sentiment-cli eval data/labeled_reviews.csv
sentiment-cli analyze data/labeled_reviews.csv
```
11. Start the web server via the CLI:
   ```bash
   sentiment-cli serve --model my_model.joblib --port 5000
   ```
12. Check the installed package version:
   ```bash
   sentiment-cli version
   # or use the global flag
   sentiment-cli --version
   ```
   You can also query the version programmatically:
   ```python
   from sentiment_analyzer_pro import __version__
   print(__version__)
   ```
13. Increase command output with the verbose flag:
   ```bash
   sentiment-cli -v train --csv data/sample_reviews.csv
   ```
14. Compare model performance (baseline vs. LSTM vs. Transformer):
   ```bash
   python -m src.model_comparison
   ```

Model comparison results are available in
[docs/MODEL_RESULTS.md](docs/MODEL_RESULTS.md).

## Deployment

To build a Docker image with all dependencies preinstalled and start the web server:

```bash
docker build -t sentiment-pro .
docker run -p 5000:5000 sentiment-pro
```

## Web API

Run a lightweight Flask server to get predictions over HTTP using the CLI:

```bash
sentiment-cli serve --model model.joblib
```

You can also invoke the underlying web app directly with the
`sentiment-web` command if preferred.

Send a POST request with JSON to `/predict`:

```bash
curl -X POST -H "Content-Type: application/json" \
     -d '{"text": "Great phone"}' http://localhost:5000/predict
```

The server will respond with the predicted sentiment.

Check that the server is running by hitting the root endpoint:

```bash
curl http://localhost:5000/
```

You should see:

```json
{"status": "ok"}
```

To verify the package version running on the server:

```bash
curl http://localhost:5000/version
```

Which will return something like:

```json
{"version": "0.1.0"}
```

## Packaging & Release

Build and publish a wheel using [build](https://pypi.org/project/build/) and
[twine](https://pypi.org/project/twine/):

```bash
pip install build twine
python -m build
twine upload dist/*
```

You can also publish automatically from GitHub by pushing a tag that starts with `v`
or by manually dispatching the workflow. The `Publish` workflow uses the
[pypa/gh-action-pypi-publish](https://github.com/pypa/gh-action-pypi-publish)
action to upload the built distributions with the `PYPI_API_TOKEN` secret.

Install the package from PyPI:

```bash
pip install sentiment-analyzer-pro
```

Optional extras provide advanced ML models and the web server:

```bash
pip install sentiment-analyzer-pro[ml,web]
```
