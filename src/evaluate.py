from sklearn.metrics import classification_report
import pandas as pd # Added for type hinting
import numpy as np # Added for type hinting

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

    Example:
        >>> y_true_labels = ['positive', 'negative', 'positive', 'negative', 'positive']
        >>> y_pred_labels = ['positive', 'positive', 'negative', 'negative', 'positive']
        >>> report = get_classification_report(y_true_labels, y_pred_labels, target_names=['negative', 'positive'])
        >>> print(report)  # doctest: +SKIP
                      precision    recall  f1-score   support
<BLANKLINE>
    negative       0.50      0.50      0.50         2
    positive       0.67      0.67      0.67         3
<BLANKLINE>
    accuracy                           0.60         5
   macro avg       0.58      0.58      0.58         5
weighted avg       0.60      0.60      0.60         5
<BLANKLINE>
    """
    try:
        # Ensure y_true and y_pred are not empty and are list-like
        if not hasattr(y_true, '__len__') or len(y_true) == 0:
            return "Error: y_true is empty or not list-like."
        if not hasattr(y_pred, '__len__') or len(y_pred) == 0:
            return "Error: y_pred is empty or not list-like."
        if len(y_true) != len(y_pred):
            return "Error: y_true and y_pred have different lengths."

        return classification_report(y_true, y_pred, **kwargs)
    except ValueError as e:
        return f"Error generating classification report: {e}. This might be due to undefined metrics for some classes."
    except Exception as e: # Catch any other unexpected errors
        return f"An unexpected error occurred while generating classification report: {e}"
