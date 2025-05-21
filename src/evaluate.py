from sklearn.metrics import classification_report
import pandas as pd # Ensure pandas is imported for Series handling
import numpy as np

# ... (get_classification_report and get_top_features functions remain) ...
def get_classification_report(y_true, y_pred, **kwargs):
    """Generates and returns a classification report from scikit-learn.

    This function wraps `sklearn.metrics.classification_report` to provide
    a standardized way to get evaluation metrics.

    Args:
        y_true (list or pd.Series or np.ndarray): Ground truth (correct) target values.
        y_pred (list or pd.Series or np.ndarray): Estimated targets as returned by a classifier.
        **kwargs: Additional keyword arguments to pass to
                  `sklearn.metrics.classification_report` (e.g., `target_names`).

    Returns:
        str: Text summary of the precision, recall, F1 score for each class.
             Returns an error message string if report generation fails.
    """
    try:
        if not hasattr(y_true, '__len__') or len(y_true) == 0:
            return "Error: y_true is empty or not list-like."
        if not hasattr(y_pred, '__len__') or len(y_pred) == 0:
            return "Error: y_pred is empty or not list-like."
        if len(y_true) != len(y_pred):
            return "Error: y_true and y_pred have different lengths."

        return classification_report(y_true, y_pred, **kwargs)
    except ValueError as e:
        return f"Error generating classification report: {e}. This might be due to undefined metrics for some classes."
    except Exception as e:
        return f"An unexpected error occurred while generating classification report: {e}"

def get_top_features(vectorizer, model_classifier, class_labels, n_top_features=10):
    """Extracts and returns the top N features for each class from a trained model.

    Applicable to models with 'coef_' (e.g., Logistic Regression, Linear SVM)
    or 'feature_log_prob_' (e.g., Naive Bayes).

    Args:
        vectorizer (sklearn.feature_extraction.text.BaseVectorizer):
            The fitted vectorizer (CountVectorizer or TfidfVectorizer).
        model_classifier (sklearn.base.BaseEstimator):
            The fitted classifier component of the pipeline
            (e.g., LogisticRegression, MultinomialNB).
        class_labels (list[str] or np.ndarray):
            The unique class labels in the order the model understands them
            (e.g., model.classes_).
        n_top_features (int, optional):
            Number of top features to return per class. Defaults to 10.

    Returns:
        dict: A dictionary where keys are class labels and values are lists
              of (feature_name, score) tuples for the top N features.
              Returns an empty dict if features cannot be extracted.
    """
    if not hasattr(vectorizer, 'get_feature_names_out'):
        print("Vectorizer does not support get_feature_names_out. Cannot extract features.")
        return {}
    
    try:
        feature_names = vectorizer.get_feature_names_out()
    except Exception as e:
        print(f"Error getting feature names: {e}")
        return {}

    top_features_per_class = {}

    if hasattr(model_classifier, 'coef_'):
        # For linear models like Logistic Regression, SVM with linear kernel
        coef = model_classifier.coef_
        
        # Handle binary classification case where coef_ shape is (1, n_features)
        # and multi-class where coef_ shape is (n_classes, n_features)
        if coef.shape[0] == 1 and len(class_labels) == 2:
            # Binary case: coef_[0] are coefficients for the class_labels[1] (second class)
            # Features for class_labels[1] (e.g., "positive")
            top_positive_indices = np.argsort(coef[0])[-n_top_features:][::-1]
            top_features_per_class[class_labels[1]] = [
                (feature_names[i], coef[0][i]) for i in top_positive_indices
            ]

            # Features for class_labels[0] (e.g., "negative") are those with the most negative coefficients
            top_negative_indices = np.argsort(coef[0])[:n_top_features]
            top_features_per_class[class_labels[0]] = [
                (feature_names[i], coef[0][i]) for i in top_negative_indices # Score is coef, not -coef
            ]
        elif coef.shape[0] == len(class_labels): # Multi-class case
            for i, class_label in enumerate(class_labels):
                class_coef = coef[i]
                top_features_indices = np.argsort(class_coef)[-n_top_features:][::-1]
                features = [(feature_names[idx], class_coef[idx]) for idx in top_features_indices]
                top_features_per_class[class_label] = features
        else:
            print(f"Coefficient shape {coef.shape} not compatible with class_labels length {len(class_labels)} for feature importance.")
            return {}

    elif hasattr(model_classifier, 'feature_log_prob_'):
        # For Naive Bayes models
        feature_log_probs = model_classifier.feature_log_prob_
        if feature_log_probs.shape[0] == len(class_labels):
            for i, class_label in enumerate(class_labels):
                class_log_probs = feature_log_probs[i]
                top_features_indices = np.argsort(class_log_probs)[-n_top_features:][::-1]
                features = [(feature_names[idx], class_log_probs[idx]) for idx in top_features_indices]
                top_features_per_class[class_label] = features
        else:
            print(f"feature_log_prob_ shape {feature_log_probs.shape} not compatible with class_labels length {len(class_labels)}.")
            return {}
    else:
        print(f"The model_classifier of type {type(model_classifier).__name__} does not have 'coef_' or 'feature_log_prob_' attributes. "
              "Cannot extract feature importance.")
        return {}

    return top_features_per_class

def show_misclassified_samples(y_true, y_pred, original_texts, n_samples=5):
    """Prints a sample of misclassified instances.

    Args:
        y_true (list, pd.Series, or np.ndarray): True labels.
        y_pred (list, pd.Series, or np.ndarray): Predicted labels.
        original_texts (list, pd.Series, or np.ndarray):
            Original text data corresponding to y_true and y_pred.
            Must be index-aligned with y_true and y_pred if they are Series.
        n_samples (int, optional):
            Maximum number of misclassified samples to display. Defaults to 5.
    """
    if not isinstance(y_true, (list, pd.Series, np.ndarray)) or \
       not isinstance(y_pred, (list, pd.Series, np.ndarray)) or \
       not isinstance(original_texts, (list, pd.Series, np.ndarray)):
        print("Error: Inputs y_true, y_pred, and original_texts must be list-like.")
        return

    if len(y_true) != len(y_pred) or len(y_true) != len(original_texts):
        print("Error: y_true, y_pred, and original_texts must have the same length.")
        return

    # Convert to pandas Series to easily find mismatches and align data using index
    # This helps if original_texts is a Series with a different index than a simple range.
    # If they are lists, they'll get a default RangeIndex.
    s_y_true = pd.Series(y_true, name="TrueLabel")
    s_y_pred = pd.Series(y_pred, name="PredictedLabel")
    s_original_texts = pd.Series(original_texts, name="OriginalText")

    # Ensure indices are aligned if inputs were lists.
    # If they were Series with meaningful indices, preserve them.
    # For this function, direct alignment by position is assumed if lists are passed.
    # If y_true, y_pred, original_texts are already pd.Series with aligned indices, this is fine.
    
    # Find indices of misclassified samples
    misclassified_indices = s_y_true[s_y_true != s_y_pred].index
    
    if misclassified_indices.empty:
        print("\nNo misclassified samples found.")
        return

    print(f"\n--- Misclassified Samples (Showing up to {n_samples}) ---")
    
    actual_n_to_show = min(n_samples, len(misclassified_indices))
    
    for i in range(actual_n_to_show):
        idx = misclassified_indices[i]
        try:
            text_to_show = s_original_texts.loc[idx]
            true_label_to_show = s_y_true.loc[idx]
            pred_label_to_show = s_y_pred.loc[idx]
            
            print(f"  Sample {i+1} (Original Index: {idx}):")
            print(f"    Original Text: \"{str(text_to_show)[:200]}...\"") # Show first 200 chars
            print(f"    True Label   : {true_label_to_show}")
            print(f"    Predicted Label: {pred_label_to_show}")
            print("-" * 20)
        except KeyError:
            # This might happen if original_texts was a list and misclassified_indices
            # refers to an index from a pd.Series that doesn't map to list index.
            # However, by converting all to Series with default range index if they are lists,
            # this should be less of an issue.
            print(f"    Could not retrieve sample at index {idx}. Indexing issue.")
            continue
    if not misclassified_indices.empty and actual_n_to_show == 0 and n_samples > 0: # This condition is unlikely if loop runs
        print("Could not display any misclassified samples due to indexing issues.")
