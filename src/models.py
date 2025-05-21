from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC # Import SVC
from sklearn.pipeline import Pipeline
import numpy as np # For predict_proba handling

class SentimentModel:
    """A sentiment analysis model that can use different classifiers and vectorizers.

    Supported classifiers: 'naive_bayes', 'logistic_regression', 'svm'.
    Supported vectorizers: 'count', 'tfidf'.

    Attributes:
        model (sklearn.pipeline.Pipeline): The scikit-learn pipeline
                                           containing the vectorizer and classifier.
        classifier_type (str): The type of classifier used.
        vectorizer_type (str): The type of vectorizer used.
    """

    def __init__(self, classifier_type='naive_bayes', vectorizer_type='count',
                 vectorizer_params=None):
        """Initializes the SentimentModel.

        Args:
            classifier_type (str, optional): Type of scikit-learn classifier.
                                             Supported: 'naive_bayes', 'logistic_regression', 'svm'.
                                             Defaults to 'naive_bayes'.
            vectorizer_type (str, optional): Type of vectorizer.
                                             Supported: 'count', 'tfidf'.
                                             Defaults to 'count'.
            vectorizer_params (dict, optional): Parameters for the vectorizer,
                                                e.g., {'ngram_range': (1, 1), 'max_features': 5000}.
                                                Defaults to None.

        Raises:
            ValueError: If an unsupported classifier_type or vectorizer_type is provided.
        """
        self.classifier_type = classifier_type
        self.vectorizer_type = vectorizer_type
        
        actual_vectorizer_params = vectorizer_params if vectorizer_params is not None else {}

        if vectorizer_type == 'count':
            vectorizer = CountVectorizer(**actual_vectorizer_params)
        elif vectorizer_type == 'tfidf':
            vectorizer = TfidfVectorizer(**actual_vectorizer_params)
        else:
            raise ValueError(f"Unsupported vectorizer_type: {vectorizer_type}. "
                             f"Supported types are 'count', 'tfidf'.")

        if classifier_type == 'naive_bayes':
            classifier = MultinomialNB()
        elif classifier_type == 'logistic_regression':
            classifier = LogisticRegression(random_state=42, solver='liblinear', max_iter=1000)
        elif classifier_type == 'svm':
            classifier = SVC(random_state=42, probability=True, kernel='linear') # Added SVM
        else:
            raise ValueError(f"Unsupported classifier_type: {classifier_type}. "
                             f"Supported types are 'naive_bayes', 'logistic_regression', 'svm'.")

        self.model = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', classifier)
        ])
        # Class docstring is updated at the class level directly now.

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
        """
        if hasattr(self.model.named_steps['classifier'], 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # This case might be less common now that SVC has probability=True
            raise AttributeError(f"The current classifier '{self.classifier_type}' ({type(self.model.named_steps['classifier']).__name__}) "
                                 "does not support predict_proba.")
