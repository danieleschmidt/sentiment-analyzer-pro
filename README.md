# Advanced Sentiment Analysis Pipeline

This project implements a comprehensive sentiment analysis pipeline, supporting various preprocessing techniques, traditional machine learning models (Naive Bayes, Logistic Regression, SVM), and Hugging Face Transformer models. It includes capabilities for hyperparameter tuning, model evaluation, feature importance analysis, misclassified sample review, and experiment tracking with MLflow.

## Project Structure

```
.
├── data/
│   └── sample_reviews.csv        # Sample CSV data for training and evaluation
├── models/                         # Default directory for saved model pickles (not for MLflow artifacts)
│   └── sentiment_model.pkl       # Example path for a saved scikit-learn pipeline
├── src/
│   ├── __init__.py
│   ├── evaluate.py               # Functions for model evaluation (classification report, top features, misclassified)
│   ├── models.py                 # scikit-learn model wrapper (SentimentModel)
│   ├── preprocessing.py          # Text preprocessing functions (basic, enhanced, NLTK downloads)
│   ├── train.py                  # Main training script with MLflow integration
│   ├── transformer_model.py      # Hugging Face Transformer model wrapper
│   └── predict_cli.py            # Command-line interface for predictions
├── tests/
│   ├── __init__.py
│   └── test_preprocessing.py     # Pytest tests for preprocessing functions
├── requirements.txt              # Project dependencies
└── README.md                     # This file
```

## Features

*   **Data Preprocessing**:
    *   Basic: Lowercasing, punctuation removal, tokenization.
    *   Enhanced: Stemming (Porter), Lemmatization (WordNet with POS tagging), configurable stopword removal (NLTK default, custom list, or none).
    *   Automatic download of NLTK resources (`punkt`, `wordnet`, `averaged_perceptron_tagger`, `stopwords`).
*   **Modeling**:
    *   **Scikit-learn**: `SentimentModel` class wrapping a `Pipeline` of a vectorizer (`CountVectorizer` or `TfidfVectorizer`) and a classifier (`MultinomialNB`, `LogisticRegression`, `SVC`).
        *   Configurable vectorizer parameters (ngram range, max features).
    *   **Hugging Face Transformers**: `TransformerSentimentModel` class for using pre-trained models (e.g., `distilbert-base-uncased-finetuned-sst-2-english`).
*   **Hyperparameter Tuning**:
    *   `GridSearchCV` for scikit-learn models with pre-defined parameter grids for common combinations.
*   **Evaluation**:
    *   Classification reports (precision, recall, F1-score).
    *   ROC AUC scores.
    *   Top feature extraction for interpretable scikit-learn models.
    *   Display of misclassified samples.
*   **Experiment Tracking**:
    *   Integration with MLflow to log parameters, metrics, and model artifacts for both scikit-learn and Transformer model runs.
*   **Prediction CLI**:
    *   Command-line interface (`src/predict_cli.py`) to load trained models (scikit-learn from `.pkl` or MLflow, Transformers from local path or MLflow) and predict sentiment for new text.

## Setup

1.  **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Create a virtual environment (recommended)**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    NLTK resources will be downloaded automatically on first import of `src.preprocessing` or when running `src.train.py` / `src.predict_cli.py`.

## Usage

### Training Models

The main script for training and evaluation is `src/train.py`. You can configure the behavior using global variables at the top of the script:

*   `CLASSIFIER_CHOICE`: `'naive_bayes'`, `'logistic_regression'`, `'svm'`, or `'transformer'`.
*   `VECTORIZER_CHOICE`: `'count'` or `'tfidf'` (for scikit-learn models).
*   `VECTORIZER_NGRAM_RANGE`, `VECTORIZER_MAX_FEATURES`: Parameters for scikit-learn vectorizers.
*   `PREPROCESSING_METHOD`: `'basic'` or `'enhanced'`.
*   `USE_STEMMING`, `USE_LEMMATIZATION`, `CUSTOM_STOPWORDS`: Options for enhanced preprocessing.
*   `PERFORM_HYPERPARAMETER_TUNING`: `True` or `False` for scikit-learn models.
*   `EVALUATE_TRANSFORMER_MODEL`: `True` or `False` to run Transformer model evaluation.
*   `TRANSFORMER_MODEL_NAME`: Name of the Hugging Face model to use.

To run training:
```bash
python src/train.py
```

This will:
*   Load and preprocess data.
*   Split data into training and test sets.
*   If `CLASSIFIER_CHOICE` is a scikit-learn type:
    *   Optionally perform hyperparameter tuning.
    *   Train the model.
    *   Save the model to `models/sentiment_model.pkl`.
    *   Evaluate and print metrics, top features, and misclassified samples.
    *   Log parameters, metrics, and the model to MLflow.
*   If `EVALUATE_TRANSFORMER_MODEL` is `True`:
    *   Load the specified Transformer model.
    *   Evaluate and print metrics and misclassified samples.
    *   Log parameters, metrics, and the model to MLflow.

### MLflow UI

To view experiment results, run the MLflow UI:
```bash
mlflow ui
```
Navigate to `http://localhost:5000` in your browser.

### Predicting with the CLI

You can use the command-line interface (`src/predict_cli.py`) to get sentiment predictions for new text using a trained model.

#### Prerequisites
- Ensure all dependencies from `requirements.txt` are installed.
- NLTK resources (`punkt`, `wordnet`, `averaged_perceptron_tagger`, `stopwords`) must be downloaded. The CLI will attempt to download them if missing.
- You need a trained model saved either as a scikit-learn pipeline (`.pkl` file or an MLflow model directory) or a Hugging Face transformer model directory.

#### Scikit-learn Model Prediction

If you have a trained scikit-learn pipeline (e.g., saved via `pickle` or logged by MLflow):

```bash
python src/predict_cli.py \
    --text "This movie was absolutely fantastic, a true masterpiece!" \
    --model_path "path/to/your/sklearn_model_dir_or_file.pkl" \
    --model_type sklearn \
    --use_lemmatization_sklearn 
    # --stopwords_sklearn path/to/custom_stopwords.txt # Optional
```
Replace `"path/to/your/sklearn_model_dir_or_file.pkl"` with the actual path. The scikit-learn model path should point to either an MLflow model directory (e.g., `mlruns/0/.../artifacts/sklearn-model/`) or a direct `.pkl` file of the **entire pipeline (preprocessor + vectorizer + classifier)**.

#### Transformer Model Prediction

If you have a saved Hugging Face transformer model (e.g., logged by MLflow or saved via `save_pretrained`):

```bash
python src/predict_cli.py \
    --text "I'm not sure how I feel about this product, it has pros and cons." \
    --model_path "path/to/your/transformer_model_dir" \
    --model_type transformer
```
Replace `"path/to/your/transformer_model_dir"` with the path to the directory containing the transformer model files (e.g., `config.json`, `pytorch_model.bin`, `tokenizer_config.json`). This could be an MLflow model directory like `mlruns/0/.../artifacts/transformer-model/`.

The CLI will print the predicted sentiment.

## Testing

To run preprocessing tests:
```bash
python -m pytest
```
(Ensure `pytest` is installed and you are in the project root directory).
