"""Trains a sentiment analysis model and evaluates it.

This script performs the following steps:
1. Loads review data from a CSV file (`data/sample_reviews.csv`).
2. Preprocesses the review text using `basic_preprocess` or `enhanced_preprocess`.
3. Splits the data into training and testing sets.
4. Optionally performs hyperparameter tuning using GridSearchCV.
5. Initializes and trains a `SentimentModel` (or uses the best tuned pipeline).
6. Saves the trained model pipeline to `models/sentiment_model.pkl`.
7. Makes predictions on the test set.
8. Prints a classification report.
"""
import pandas as pd
from src.preprocessing import basic_preprocess, enhanced_preprocess
from src.models import SentimentModel # Our wrapper
from src.transformer_model import TransformerSentimentModel # Import the transformer model
from src.evaluate import get_classification_report, get_top_features, show_misclassified_samples # Added show_misclassified_samples
from sklearn.metrics import roc_auc_score # Import roc_auc_score
import numpy as np # For label binarization
from sklearn.model_selection import train_test_split, GridSearchCV # Added GridSearchCV
from sklearn.pipeline import Pipeline as SklearnPipeline # Alias for scikit-learn's Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pickle
import os

# Define file paths
DATA_PATH = 'data/sample_reviews.csv'
MODEL_PATH = 'models/sentiment_model.pkl'

# --- Global Model Configuration ---
CLASSIFIER_CHOICE = 'svm' # 'naive_bayes', 'logistic_regression', 'svm'
# Set to 'transformer' to only run the transformer model, or any other sklearn model type
# CLASSIFIER_CHOICE = 'transformer' # Example to exclusively test transformer

VECTORIZER_CHOICE = 'tfidf'     # 'count' or 'tfidf'
VECTORIZER_NGRAM_RANGE = (1, 1) # e.g., (1,1) for unigrams, (1,2) for unigrams and bigrams
VECTORIZER_MAX_FEATURES = None  # e.g., 5000 to limit feature space, None for no limit

# --- Global Preprocessing Configuration ---
PREPROCESSING_METHOD = 'enhanced' # 'basic' or 'enhanced'
USE_STEMMING = False
USE_LEMMATIZATION = True # Lemmatization is often preferred over stemming
CUSTOM_STOPWORDS = None # None for NLTK default, [] for no stopwords, or ['list', 'of', 'words']

# --- Configuration for Hyperparameter Tuning ---
PERFORM_HYPERPARAMETER_TUNING = True # Set to False to skip tuning for scikit-learn models
SAMPLE_PARAM_GRID = {
    'tfidf_logreg': {
        'vectorizer__ngram_range': [(1, 1), (1, 2)],
        'vectorizer__max_features': [2500, 5000],
        'classifier__C': [0.1, 1, 10]
    },
    'count_naive_bayes': {
        'vectorizer__ngram_range': [(1, 1), (1, 2)],
        # 'vectorizer__max_features': [2500, 5000], # Example: Naive Bayes might not always benefit from max_features
        'classifier__alpha': [0.1, 0.5, 1.0]
    },
    'tfidf_svm': {
        'vectorizer__ngram_range': [(1,1), (1,2)],
        'vectorizer__max_df': [0.90, 0.95, 1.0], # Max document frequency
        'vectorizer__min_df': [1, 2, 5],       # Min document frequency
        'vectorizer__max_features': [None, 5000, 10000], # Number of features
        'classifier__C': [0.1, 1, 10],
        'classifier__kernel': ['linear'] # Keep linear for now, rbf can be very slow
    }
    # Add more grids for other combinations as needed (e.g., 'count_svm', 'tfidf_naive_bayes')
}

def tune_sentiment_model_hyperparameters(X_train, y_train, classifier_type, vectorizer_type,
                                         base_vectorizer_params, param_grid_key):
    """Tunes hyperparameters for a sentiment model pipeline using GridSearchCV.

    Args:
        X_train (pd.Series): Training text data.
        y_train (pd.Series): Training labels.
        classifier_type (str): Type of classifier ('naive_bayes', 'logistic_regression', 'svm').
        vectorizer_type (str): Type of vectorizer ('count', 'tfidf').
        base_vectorizer_params (dict): Base parameters for the vectorizer.
                                       These will be fixed unless overridden by param_grid.
        param_grid_key (str): Key to select the parameter grid from SAMPLE_PARAM_GRID.

    Returns:
        sklearn.pipeline.Pipeline: The best fitted pipeline from GridSearchCV.
                                   Returns None if tuning is skipped or fails.
    """
    print(f"\nAttempting hyperparameter tuning for {vectorizer_type} with {classifier_type}...")

    # Select vectorizer
    if vectorizer_type == 'count':
        vectorizer = CountVectorizer(**base_vectorizer_params)
    elif vectorizer_type == 'tfidf':
        vectorizer = TfidfVectorizer(**base_vectorizer_params)
    else:
        raise ValueError(f"Unsupported vectorizer_type: {vectorizer_type}")

    # Select classifier
    if classifier_type == 'naive_bayes':
        classifier = MultinomialNB()
    elif classifier_type == 'logistic_regression':
        classifier = LogisticRegression(random_state=42, solver='liblinear', max_iter=1000)
    elif classifier_type == 'svm':
        classifier = SVC(random_state=42, probability=True)
    else:
        raise ValueError(f"Unsupported classifier_type: {classifier_type}")

    pipeline = SklearnPipeline([
        ('vectorizer', vectorizer),
        ('classifier', classifier)
    ])

    if param_grid_key not in SAMPLE_PARAM_GRID:
        print(f"Warning: Parameter grid for '{param_grid_key}' not found in SAMPLE_PARAM_GRID. Skipping tuning.")
        return None
    
    current_param_grid = SAMPLE_PARAM_GRID[param_grid_key]
    print(f"Using parameter grid: {current_param_grid}")

    try:
        grid_search = GridSearchCV(pipeline, current_param_grid, cv=3, scoring='accuracy', verbose=1, n_jobs=-1)
        print("Starting GridSearchCV... This may take a while.")
        grid_search.fit(X_train, y_train)

        print("\n--- Hyperparameter Tuning Results ---")
        print(f"Best parameters found: {grid_search.best_params_}")
        print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")
        print("------------------------------------")
        return grid_search.best_estimator_
    except Exception as e:
        print(f"Error during GridSearchCV: {e}")
        print("Skipping hyperparameter tuning due to error.")
        return None


def main():
    """Main function to run the training and evaluation pipeline."""
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    print(f"Using model directory: {os.path.dirname(MODEL_PATH)}")

    print(f"Loading data from {DATA_PATH}...")
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}")
        print(f"Creating a dummy {DATA_PATH} for demonstration purposes.")
        dummy_data = {'review': ["good", "bad", "excellent", "terrible", "not bad", "amazing"] * 10, 
                      'sentiment': ["positive", "negative", "positive", "negative", "positive", "positive"] * 10}
        dummy_df = pd.DataFrame(dummy_data)
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        dummy_df.to_csv(DATA_PATH, index=False)
        df = dummy_df
    else:
        df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} reviews.")

    print("Preprocessing data...")
    if PREPROCESSING_METHOD == 'enhanced':
        print(f"Using enhanced_preprocess with stemming={USE_STEMMING}, lemmatization={USE_LEMMATIZATION}, custom_stopwords_list type={type(CUSTOM_STOPWORDS)}")
        df['processed_review'] = df['review'].astype(str).apply(
            lambda x: ' '.join(enhanced_preprocess(
                x, use_stemming=USE_STEMMING, use_lemmatization=USE_LEMMATIZATION,
                custom_stopwords_list=CUSTOM_STOPWORDS
            ))
        )
    else:
        print("Using basic_preprocess.")
        df['processed_review'] = df['review'].astype(str).apply(lambda x: ' '.join(basic_preprocess(x)))
    print("Preprocessing complete.")

    X = df['processed_review'] # Processed text for sklearn models
    y = df['sentiment']
    X_original_text = df['review'] # Original raw text

    stratify_option = None
    if len(y.unique()) > 1:
        class_counts = y.value_counts()
        if all(count >= 2 for count in class_counts):
             stratify_option = y
        else:
            print("Warning: Not all classes have enough samples to stratify. Proceeding without stratification.")
            # stratify_option remains None
            
    X_train, X_test, y_train, y_test, X_train_original_for_analysis, X_test_original_for_analysis = train_test_split(
        X, y, X_original_text, # Include original text in split
        test_size=0.2, 
        random_state=42, 
        stratify=stratify_option
    )
    print(f"Data split into {len(X_train)} training samples and {len(X_test)} test samples.")
    print(f"Original text for test set preserved: {len(X_test_original_for_analysis)} samples.")

    # --- Scikit-learn Model Processing ---
    if CLASSIFIER_CHOICE != 'transformer':
        print("\n" + "="*50)
        print("SCIKIT-LEARN MODEL PROCESSING")
        print("="*50)
        print(f"\nInitializing SentimentModel with base configuration: "
              f"Classifier='{CLASSIFIER_CHOICE}', "
              f"Vectorizer='{VECTORIZER_CHOICE}', "
              f"NgramRange={VECTORIZER_NGRAM_RANGE}, "
              f"MaxFeatures={VECTORIZER_MAX_FEATURES}")

        base_vectorizer_params_config = {
            'ngram_range': VECTORIZER_NGRAM_RANGE,
        }
        base_vectorizer_params_config = {k: v for k, v in base_vectorizer_params_config.items() if v is not None}

        param_grid_key_for_tuning = f"{VECTORIZER_CHOICE}_{CLASSIFIER_CHOICE.replace('_', '')}" 
        
        if VECTORIZER_MAX_FEATURES is not None:
            is_max_features_in_tuning_grid = False
            if PERFORM_HYPERPARAMETER_TUNING and param_grid_key_for_tuning in SAMPLE_PARAM_GRID:
                if any('vectorizer__max_features' in k for k in SAMPLE_PARAM_GRID[param_grid_key_for_tuning]):
                    is_max_features_in_tuning_grid = True
            if not is_max_features_in_tuning_grid:
                base_vectorizer_params_config['max_features'] = VECTORIZER_MAX_FEATURES

        best_model_pipeline = None
        if PERFORM_HYPERPARAMETER_TUNING:
            tuned_pipeline = tune_sentiment_model_hyperparameters(
                X_train, y_train,
                CLASSIFIER_CHOICE, VECTORIZER_CHOICE,
                base_vectorizer_params_config.copy(),
                param_grid_key_for_tuning
            )
            if tuned_pipeline:
                best_model_pipeline = tuned_pipeline
                print("Using tuned model pipeline for training and prediction.")
        
        model_to_train_and_evaluate = None
        if best_model_pipeline:
            model_to_train_and_evaluate = best_model_pipeline
            print("Tuned model is already fitted. Proceeding to evaluation.")
        else:
            print("Proceeding with base model configuration (or tuning was skipped/failed).")
            final_vectorizer_params = base_vectorizer_params_config.copy()
            if VECTORIZER_MAX_FEATURES is not None and 'max_features' not in final_vectorizer_params:
                 final_vectorizer_params['max_features'] = VECTORIZER_MAX_FEATURES
            
            regular_model_wrapper = SentimentModel(
                classifier_type=CLASSIFIER_CHOICE,
                vectorizer_type=VECTORIZER_CHOICE,
                vectorizer_params=final_vectorizer_params
            )
            print("Training base model...")
            regular_model_wrapper.train(X_train, y_train)
            model_to_train_and_evaluate = regular_model_wrapper.model
            print("Base model trained successfully.")

        print(f"\nSaving scikit-learn model to {MODEL_PATH}...")
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(model_to_train_and_evaluate, f)
        print(f"Trained scikit-learn model (pipeline) saved to {MODEL_PATH}")

        if not X_test.empty:
            print("\nMaking predictions on the test set with scikit-learn model...")
            y_pred = model_to_train_and_evaluate.predict(X_test)

            print("\nScikit-learn Model - Sample Test Set Predictions:")
            sample_display_limit = min(5, len(X_test))
            for i in range(sample_display_limit):
                original_review_text = df.loc[X_test.index[i], 'review'] # Use original df index
                print(f"  Review: {original_review_text[:60]}... | Actual: {y_test.iloc[i]} | Predicted: {y_pred[i]}")

            print("\nScikit-learn Model - Classification Report on Test Set:")
            try:
                report = get_classification_report(y_test, y_pred, target_names=sorted(list(y.unique())))
                print(report)
            except ValueError as e:
                print(f"Could not generate classification report for scikit-learn model: {e}")
            except Exception as e:
                print(f"An unexpected error occurred during scikit-learn model report: {e}")

            if hasattr(model_to_train_and_evaluate, 'named_steps'):
                final_vectorizer = model_to_train_and_evaluate.named_steps.get('vectorizer')
                final_classifier = model_to_train_and_evaluate.named_steps.get('classifier')

                if final_vectorizer and final_classifier and \
                   (hasattr(final_classifier, 'coef_') or hasattr(final_classifier, 'feature_log_prob_')):
                    print("\n--- Top Features per Class (Scikit-learn Model) ---")
                    try:
                        model_class_labels = final_classifier.classes_
                        top_features_data = get_top_features(
                            vectorizer=final_vectorizer,
                            model_classifier=final_classifier,
                            class_labels=model_class_labels,
                            n_top_features=10
                        )
                        if top_features_data:
                            for label, features in top_features_data.items():
                                print(f"  Top features for class '{label}':")
                                for feature, score in features:
                                    print(f"    - {feature}: {score:.4f}")
                        else:
                            print("  Could not extract top features for the scikit-learn model.")
                    except Exception as e_feat:
                        print(f"  Error extracting top features: {e_feat}")
                else:
                    print("\nSkipping feature importance for the scikit-learn model: "
                          "Model type might not be supported or pipeline structure is unexpected.")

            print("\n--- Misclassified Samples (Scikit-learn Model) ---")
            show_misclassified_samples(
                y_test, 
                y_pred, 
                X_test_original_for_analysis.to_list(), # Pass as list to match default indexing of y_test/y_pred if they are numpy arrays
                n_samples=5
            )
        else:
            print("\nTest set is empty. Skipping scikit-learn model predictions, feature analysis, and misclassified samples.")
        
        print("\n" + "="*50)
        print("SCIKIT-LEARN MODEL EVALUATION COMPLETE")
        print("="*50 + "\n")
    else:
        print("Skipping scikit-learn model processing as CLASSIFIER_CHOICE is 'transformer'.")

    # --- Transformer Model Evaluation ---
    if EVALUATE_TRANSFORMER_MODEL:
        print("\n" + "="*50)
        print("TRANSFORMER MODEL EVALUATION")
        print("="*50)
        
        print(f"Initializing TransformerSentimentModel with model: {TRANSFORMER_MODEL_NAME}...")
        transformer_model_instance = TransformerSentimentModel(model_name=TRANSFORMER_MODEL_NAME)

        if transformer_model_instance.model:
            transformer_model_instance.train(None, None) 

            if not X_test.empty: # X_test here refers to processed text, but we need original for transformer
                try:
                    # Use X_test_original_for_analysis which holds original text for the test split
                    original_texts_for_transformer_test = X_test_original_for_analysis.tolist()
                    y_true_for_transformer = y_test.tolist() # y_test is already aligned with X_test_original_for_analysis

                    print(f"Predicting with Transformer model (and getting probabilities) on {len(original_texts_for_transformer_test)} test samples...")
                    transformer_output = transformer_model_instance.predict(
                        original_texts_for_transformer_test,
                        return_probabilities=True
                    )
                    
                    if not transformer_output or len(transformer_output) != 2:
                        print("Error: Transformer model predict() did not return expected tuple (labels, probabilities).")
                        transformer_y_pred_labels, transformer_y_pred_positive_probs = [], []
                    else:
                        transformer_y_pred_labels, transformer_y_pred_positive_probs = transformer_output

                    print("\nTransformer Model - Sample Test Set Predictions (with Probs):")
                    sample_display_limit = min(5, len(original_texts_for_transformer_test))
                    for i in range(sample_display_limit):
                        prob_display = f"{transformer_y_pred_positive_probs[i]:.4f}" if i < len(transformer_y_pred_positive_probs) else "N/A"
                        label_display = transformer_y_pred_labels[i] if i < len(transformer_y_pred_labels) else "N/A"
                        print(f"  Review: {original_texts_for_transformer_test[i][:60]}... | Actual: {y_true_for_transformer[i]} | Predicted: {label_display} | Pos_Prob: {prob_display}")

                    print("\nTransformer Model - Classification Report on Test Set:")
                    try:
                        if transformer_y_pred_labels:
                             report_transformer = get_classification_report(y_true_for_transformer, transformer_y_pred_labels, target_names=sorted(list(set(y_true_for_transformer)|set(transformer_y_pred_labels))))
                             print(report_transformer)
                        else:
                            print("Skipping classification report as no predictions were made.")
                    except ValueError as e:
                        print(f"Could not generate classification report for Transformer: {e}")
                    except Exception as e_gen:
                        print(f"An unexpected error occurred during Transformer classification report: {e_gen}")

                    if transformer_y_pred_positive_probs and y_true_for_transformer:
                        try:
                            y_true_binarized_for_roc = np.array([1 if label.lower() == 'positive' else 0 for label in y_true_for_transformer])
                            if len(np.unique(y_true_binarized_for_roc)) > 1:
                                roc_auc = roc_auc_score(y_true_binarized_for_roc, transformer_y_pred_positive_probs)
                                print(f"Transformer Model - ROC AUC Score: {roc_auc:.4f}")
                            else:
                                print("ROC AUC score not calculated because only one class is present in the true labels of the test set.")
                        except ValueError as ve:
                            print(f"Could not calculate ROC AUC score: {ve}. This might happen if all predictions are of a single class or probabilities are not varied.")
                        except Exception as e_roc:
                            print(f"An unexpected error occurred during ROC AUC calculation: {e_roc}")
                    else:
                        print("Skipping ROC AUC calculation due to empty predictions or true labels for transformer.")

                    if transformer_y_pred_labels:
                        print("\n--- Misclassified Samples (Transformer Model) ---")
                        show_misclassified_samples(
                            y_test, 
                            transformer_y_pred_labels, 
                            X_test_original_for_analysis.to_list(), # Pass as list
                            n_samples=5
                        )
                    else:
                        print("Skipping misclassified samples for Transformer as no predictions were made/available.")

                except KeyError as e:
                    print(f"KeyError: Could not retrieve original texts for transformer. Error: {e}")
                except Exception as e_general:
                     print(f"An unexpected error occurred during Transformer model evaluation: {e_general}")
            else:
                print("Test set is empty. Skipping Transformer model predictions.")
        else:
            print("Transformer model failed to load. Skipping its evaluation.")
        print("\n" + "="*50)
        print("TRANSFORMER MODEL EVALUATION COMPLETE")
        print("="*50 + "\n")
    else:
        print("Skipping Transformer model evaluation as per configuration.")

    print("-" * 50)
    print("Script finished.")

if __name__ == '__main__':
    EVALUATE_TRANSFORMER_MODEL = True 
    TRANSFORMER_MODEL_NAME = 'distilbert-base-uncased-finetuned-sst-2-english'
    main()
