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

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
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
7. Compare model performance (baseline vs. LSTM vs. Transformer):
   ```bash
   python -m src.model_comparison
   ```

Model comparison results are available in
[docs/MODEL_RESULTS.md](docs/MODEL_RESULTS.md).
