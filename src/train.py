"""Trains a sentiment analysis model and evaluates it.

This script performs the following steps:
1. Loads review data from a CSV file (`data/sample_reviews.csv`).
2. Preprocesses the review text using `basic_preprocess`.
3. Splits the data into training and testing sets.
4. Initializes and trains a `SentimentModel` (defaulting to Naive Bayes).
5. Saves the trained model to `models/sentiment_model.pkl`.
6. Makes predictions on the test set.
7. Prints a classification report.
"""
import pandas as pd
from src.preprocessing import basic_preprocess
from src.models import SentimentModel
from src.evaluate import get_classification_report
from sklearn.model_selection import train_test_split
import pickle
import os

# Define file paths
DATA_PATH = 'data/sample_reviews.csv'
MODEL_PATH = 'models/sentiment_model.pkl'
# Alternative model type, e.g., 'logistic_regression'
# CLASSIFIER_CHOICE = 'logistic_regression'
CLASSIFIER_CHOICE = 'naive_bayes'


def main():
    """Main function to run the training and evaluation pipeline."""
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    print(f"Using model directory: {os.path.dirname(MODEL_PATH)}")

    # Load data
    print(f"Loading data from {DATA_PATH}...")
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}")
        # Create a dummy file for the script to run if it doesn't exist
        print(f"Creating a dummy {DATA_PATH} for demonstration purposes.")
        dummy_data = {'review': ["good", "bad", "excellent", "terrible", "not bad", "amazing"], 'sentiment': ["positive", "negative", "positive", "negative", "positive", "positive"]}
        dummy_df = pd.DataFrame(dummy_data)
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True) # Ensure data directory exists
        dummy_df.to_csv(DATA_PATH, index=False)
        df = dummy_df
    else:
        df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} reviews.")

    # Preprocess data
    print("Preprocessing data...")
    # Ensure 'review' column is of string type before applying basic_preprocess
    df['processed_review'] = df['review'].astype(str).apply(
        lambda x: ' '.join(basic_preprocess(x))
    )
    print("Preprocessing complete.")

    # Prepare data for training
    X = df['processed_review']
    y = df['sentiment']

    # Stratify only if there are enough samples for each class to be split
    stratify_option = None
    if len(y.unique()) > 1: # Check if there is more than one class
        # Check if all classes have at least 2 members for a 0.2 test split (needs at least 1 for train, 1 for test)
        # For a test_size of 0.2, a class needs at least 2 samples to guarantee one in each split.
        # More generally, for n_splits in StratifiedKFold (which train_test_split uses internally for stratification),
        # each class must have at least n_splits members. Here, it implies at least 2 for a single split.
        class_counts = y.value_counts()
        if all(count >= 2 for count in class_counts):
             stratify_option = y
        else:
            print("Warning: Not all classes have enough samples to stratify. Proceeding without stratification.")

    # Split data into training and testing sets (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify_option)
    print(f"Data split into {len(X_train)} training samples and {len(X_test)} test samples.")

    # Train model
    print(f"Initializing SentimentModel with classifier_type='{CLASSIFIER_CHOICE}'...")
    model = SentimentModel(classifier_type=CLASSIFIER_CHOICE)
    print("Training model...")
    model.train(X_train, y_train)
    print(f"Model trained successfully on the training data.")

    # Save model
    print(f"Saving model to {MODEL_PATH}...")
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    print(f"Trained model saved to {MODEL_PATH}")

    # Predict on the test set
    if not X_test.empty:
        print("\nMaking predictions on the test set...")
        y_pred = model.predict(X_test)

        # Print some predictions for qualitative check
        print("\nSample Test Set Predictions:")
        sample_display_limit = min(5, len(X_test))
        for i in range(sample_display_limit):
            original_review_text = df.loc[X_test.index[i], 'review'] # Use original index from X_test
            print(f"  Review: {original_review_text[:60]}... | Actual: {y_test.iloc[i]} | Predicted: {y_pred[i]}")

        # Print classification report
        print("\nClassification Report on Test Set:")
        try:
            # Ensure target_names are correctly ordered if using them
            # For binary classification, if your labels are 'positive' and 'negative',
            # and you want a specific order in the report, you can pass target_names.
            # Example: target_names=sorted(list(y.unique()))
            report = get_classification_report(y_test, y_pred, target_names=sorted(list(y.unique())))
            print(report)
        except ValueError as e:
            print(f"Could not generate classification report: {e}")
            print("This can happen if the test set is too small or has only one class after splitting.")
        except Exception as e: # Catch other potential errors from classification_report
            print(f"An unexpected error occurred while generating the classification report: {e}")

    else:
        print("\nTest set is empty. Skipping predictions and classification report.")
    print("-" * 50)
    print("Script finished.")

if __name__ == '__main__':
    main()
