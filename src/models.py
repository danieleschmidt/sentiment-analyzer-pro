from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression # Added for flexibility
from sklearn.pipeline import Pipeline
import numpy as np # Added for type hinting

class SentimentModel:
    """A sentiment analysis model that can use different classifiers.

    Attributes:
        model (sklearn.pipeline.Pipeline): The scikit-learn pipeline
                                           containing the vectorizer and classifier.
        classifier_type (str): The type of classifier used by the model.
    """

    def __init__(self, classifier_type='naive_bayes'):
        """Initializes the SentimentModel.

        Args:
            classifier_type (str, optional): The type of scikit-learn classifier to use.
                                             Supported types: 'naive_bayes', 'logistic_regression'.
                                             Defaults to 'naive_bayes'.

        Raises:
            ValueError: If an unsupported classifier_type is provided.
        """
        if classifier_type == 'naive_bayes':
            classifier = MultinomialNB()
        elif classifier_type == 'logistic_regression':
            # You might want to add solver and other params for LogisticRegression
            classifier = LogisticRegression(random_state=42, solver='liblinear')
        else:
            raise ValueError(f"Unsupported classifier_type: {classifier_type}. "
                             f"Supported types are 'naive_bayes', 'logistic_regression'.")

        self.model = Pipeline([
            ('vectorizer', CountVectorizer()),
            ('classifier', classifier)
        ])
        self.classifier_type = classifier_type # Store for reference if needed

    def train(self, X, y):
        """Trains the sentiment analysis model.

        Args:
            X (list or pd.Series): A list or pandas Series of text samples.
            y (list or pd.Series): A list or pandas Series of corresponding sentiment labels.
        """
        self.model.fit(X, y)

    def predict(self, X):
        """Predicts sentiment for new text samples.

        Args:
            X (list or pd.Series): A list or pandas Series of text samples.

        Returns:
            np.ndarray: An array of predicted sentiment labels.
        """
        return self.model.predict(X)

    def predict_proba(self, X):
        """Predicts sentiment probabilities for new text samples.

        Args:
            X (list or pd.Series): A list or pandas Series of text samples.

        Returns:
            np.ndarray: An array of predicted sentiment probabilities
                        (shape: [n_samples, n_classes]).
        
        Raises:
            AttributeError: If the classifier does not support probability prediction.
        """
        if hasattr(self.model.named_steps['classifier'], 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # For classifiers that don't have predict_proba (like SGDClassifier without loss='log_loss' or 'modified_huber')
            # you might return None or raise an error, or return a one-hot encoded prediction.
            # For simplicity, let's raise an error here.
            raise AttributeError(f"The current classifier '{self.classifier_type}'"
                                 " does not support predict_proba.")
