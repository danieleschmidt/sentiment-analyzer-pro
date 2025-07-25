# Sentiment Analyzer Pro

This project aims to develop an advanced sentiment analysis tool. It will start with a basic sentiment classifier and progressively add features like aspect-based sentiment analysis, sophisticated text preprocessing, model comparison, and robust evaluation.

## Project Goals
- ✅ Implement and compare various sentiment analysis models (Naive Bayes, Logistic Regression, RNN/LSTM, **Transformer-based**).
- ✅ Develop robust text preprocessing pipelines.
- ✅ Explore aspect-based sentiment analysis.
- ✅ Implement comprehensive model evaluation and error analysis.
- ✅ **Advanced BERT fine-tuning pipeline with full training capabilities**.
- ✅ **Comprehensive model comparison framework with detailed performance metrics**.
- ✅ Ensure code is well-documented and tested.

## Quick Start

1. **Set up the environment:**
   ```bash
   make setup
   ```

2. **Start the web server:**
   ```bash
   python -m src.webapp
   ```

3. **Make your first prediction:**
   ```bash
   curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "I love this product!"}'
   ```

4. **View comprehensive documentation:**
   - [Getting Started Guide](docs/GETTING_STARTED.md) - Complete setup and usage guide
   - [API Reference](docs/API_REFERENCE.md) - Detailed API documentation

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
4. Build a Naive Bayes model:
   ```python
   from src.models import build_nb_model
   nb_model = build_nb_model()
   ```
5. (Optional) Build the LSTM model for experimentation:
   ```python
   from src.models import build_lstm_model
   model = build_lstm_model()
   ```
6. **NEW: Advanced Transformer Training & Comparison**:
   ```python
   # Quick model comparison (fast)
   from src.model_comparison import benchmark_models
   results = benchmark_models("data/sample_reviews.csv", include_transformer_training=False)
   
   # Full BERT fine-tuning (requires torch + transformers)
   from src.transformer_trainer import TransformerTrainer, TransformerConfig
   config = TransformerConfig(num_epochs=3, batch_size=16)
   trainer = TransformerTrainer(config)
   results = trainer.train("data/sample_reviews.csv")
   
   # Comprehensive model comparison with training
   results = benchmark_models("data/sample_reviews.csv", include_transformer_training=True)
   ```
7. Make predictions on a CSV file with a `text` column:
   ```bash
   python -m src.predict your_reviews.csv
   ```
8. Preprocess a CSV of raw reviews (with optional lemmatization):
   ```bash
   sentiment-cli preprocess data/raw.csv --out clean.csv --remove-stopwords --lemmatize
   ```
9. Split a labeled dataset into train and test files:
   ```bash
   sentiment-cli split data/all.csv --train train.csv --test test.csv --ratio 0.2
   ```
10. Summarize a dataset with basic statistics:
   ```bash
sentiment-cli summary train.csv
# show the five most common words
sentiment-cli summary train.csv --top 5
```
The `--top` flag lists the most frequent tokens in the dataset. Punctuation is
automatically stripped when counting words, so `movie!` and `movie.` are
aggregated under the same entry.
11. Use the unified CLI for training, prediction, evaluation, and analysis:
   ```bash
sentiment-cli train --csv data/sample_reviews.csv --model my_model.joblib
sentiment-cli predict your_reviews.csv --model my_model.joblib
# Both commands also read the MODEL_PATH environment variable when --model
# is omitted.
sentiment-cli eval data/labeled_reviews.csv
sentiment-cli analyze data/labeled_reviews.csv
```
12. Start the web server via the CLI (binds to `127.0.0.1` by default):
   ```bash
   sentiment-cli serve --model my_model.joblib --port 5000
   ```
   To expose it to other machines, pass `--host 0.0.0.0` (use with caution).
13. Check the installed package version:
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
14. Increase command output with the verbose flag:
   ```bash
   sentiment-cli -v train --csv data/sample_reviews.csv
   ```
15. Compare model performance (baseline vs. LSTM vs. Transformer):
   ```bash
   python -m src.model_comparison
   ```
16. Evaluate a model using stratified k-fold cross-validation:
   ```python
   from src.evaluate import cross_validate
   import pandas as pd

   data = pd.read_csv("data/sample_reviews.csv")
   score = cross_validate(data["text"], data["label"], folds=5)
   print(f"CV accuracy: {score:.2f}")  # texts and labels must be the same length
   # Use a different model:
   from src.models import build_nb_model
   nb_score = cross_validate(
       data["text"],
       data["label"],
       folds=5,
       model_fn=build_nb_model,
   )
   print(f"NB CV accuracy: {nb_score:.2f}")
   # cross_validate uses StratifiedKFold to keep label ratios consistent
   ```
17. Run stratified cross-validation from the CLI:
   ```bash
   sentiment-cli crossval data/sample_reviews.csv --folds 3
   ```
   Use macro F1 instead of accuracy:
   ```bash
   sentiment-cli crossval data/sample_reviews.csv --metric f1
   ```

Model comparison results are available in
[docs/MODEL_RESULTS.md](docs/MODEL_RESULTS.md).

## 🚀 Advanced Transformer Capabilities

The project now includes state-of-the-art transformer model support with comprehensive training and evaluation capabilities:

### Quick Model Comparison
```python
from src.model_comparison import benchmark_models

# Fast comparison (no transformer training)
results = benchmark_models("data/sample_reviews.csv", include_transformer_training=False)
```

### Full BERT Fine-tuning
```python
from src.transformer_trainer import TransformerTrainer, TransformerConfig

# Configure transformer training
config = TransformerConfig(
    model_name="distilbert-base-uncased",
    num_epochs=3,
    batch_size=16,
    learning_rate=2e-5,
    max_length=128,
    output_dir="models/my_sentiment_model"
)

# Train the model
trainer = TransformerTrainer(config)
results = trainer.train("data/sample_reviews.csv")

# Make predictions
predictions = trainer.predict(["Great product!", "Terrible experience"])
print(predictions)  # ['positive', 'negative']
```

### Comprehensive Model Comparison
```python
from src.model_comparison import ComprehensiveModelComparison

# Detailed comparison with metrics
comparison = ComprehensiveModelComparison("data/sample_reviews.csv")
results = comparison.compare_all_models(include_transformer_training=True)
comparison.print_comparison_table()

# Get results as DataFrame
df = comparison.get_results_dataframe()
```

### Example Usage
Run the complete example script:
```bash
python examples/transformer_example.py
```

### Performance Metrics
The framework provides detailed metrics for each model:
- **Accuracy, Precision, Recall, F1-Score**: Standard classification metrics
- **Training Time**: Time to train the model
- **Prediction Time**: Average inference time per sample
- **Additional Metrics**: Model-specific information (loss, parameters, etc.)

### Requirements for Full Functionality
For complete transformer capabilities, install the optional ML dependencies:
```bash
pip install torch transformers
# or
pip install sentiment-analyzer-pro[ml]
```

The framework gracefully handles missing dependencies and provides reasonable baseline estimates when transformer libraries are not available.

## Deployment

To build a Docker image with all dependencies preinstalled and start the web server:

```bash
docker build -t sentiment-pro .
docker run -p 5000:5000 sentiment-pro
# add --host 0.0.0.0 if you need external access
```

## Web API

Run a lightweight Flask server to get predictions over HTTP using the CLI:

```bash
sentiment-cli serve --model model.joblib
```
By default the server binds to `127.0.0.1` for safety. Use the
`--host 0.0.0.0` option only when you need external access.

You can also invoke the underlying web app directly with the
`sentiment-web` command if preferred.

The server will also read the `MODEL_PATH` environment variable as the default
model location if `--model` is not supplied.

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

You can fetch simple service metrics:

```bash
curl http://localhost:5000/metrics
```
Which returns JSON similar to:

```json
{"requests": 42, "predictions": 40}
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

## Documentation

### User Guides
- **[Getting Started](docs/GETTING_STARTED.md)** - Complete setup and tutorial guide
- **[API Reference](docs/API_REFERENCE.md)** - REST API and CLI documentation
- **[Security Guide](docs/SECURITY.md)** - Security features and best practices

### Technical Documentation
- [Data Handling](docs/DATA_HANDLING.md) - Data processing and validation
- [Model Evaluation](docs/EVALUATION.md) - Model performance evaluation
- [Model Results](docs/MODEL_RESULTS.md) - Comparative analysis of different models
- [Aspect-based Sentiment](docs/ASPECT_SENTIMENT.md) - Advanced sentiment analysis features

### Development
- [Development Plan](DEVELOPMENT_PLAN.md) - Roadmap and architecture
- [Contributing Guidelines](CONTRIBUTING.md) - How to contribute
- [Code Review Guidelines](CODE_REVIEW.md) - Code review process

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions and pull request
guidelines. After installing dependencies run `pre-commit install` to enable
automatic linting and secret scanning. Always run `pytest -q` before submitting
changes.

## License

This project is licensed under the [MIT License](LICENSE).
