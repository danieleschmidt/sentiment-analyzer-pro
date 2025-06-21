# Data Handling Guide

This project uses CSV files for training and prediction. Each CSV must contain a `text` column with the review text and a `label` column with sentiment labels for supervised learning.

Sample dataset is located at `data/sample_reviews.csv`. To add new data, place additional CSV files in the `data/` directory. Ensure that text is preprocessed using the utilities in `src/preprocessing.py` before training advanced models. For Transformer-based models, tokenization should be performed with the Hugging Face `transformers` library (see `build_transformer_model` in `src/models.py`).
