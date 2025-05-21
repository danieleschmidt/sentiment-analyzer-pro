from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch # PyTorch is a dependency for transformers

class TransformerSentimentModel:
    """A sentiment analysis model using Hugging Face Transformers.

    This model uses a pre-trained transformer model for sentiment analysis.
    It can perform zero-shot prediction. Fine-tuning is not implemented
    in this basic version but can be added.

    Attributes:
        model_name (str): The name of the pre-trained Hugging Face model.
        sentiment_pipeline (transformers.pipelines.Pipeline): The sentiment analysis pipeline.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer for the model.
        model (transformers.PreTrainedModel): The loaded transformer model.
    """

    def __init__(self, model_name='distilbert-base-uncased-finetuned-sst-2-english'):
        """Initializes the TransformerSentimentModel.

        Args:
            model_name (str, optional): The name of the pre-trained Hugging Face model
                                        to use. Defaults to
                                        'distilbert-base-uncased-finetuned-sst-2-english'.
                                        This model typically outputs 'POSITIVE' or 'NEGATIVE'.
        """
        self.model_name = model_name
        try:
            # Using a pipeline is simpler for direct sentiment classification
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                # device=0 if torch.cuda.is_available() else -1 # Use GPU if available
            )
            # For more control, you can load tokenizer and model separately
            # self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            # self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            print(f"Transformer model '{self.model_name}' loaded successfully.")
        except Exception as e:
            print(f"Error loading transformer model '{self.model_name}': {e}")
            print("Please ensure you have an internet connection and the model name is correct.")
            print("You might need to install additional dependencies like 'torch' or 'tensorflow'.")
            self.sentiment_pipeline = None # Indicate failure

    def train(self, X, y, **kwargs):
        """Placeholder for training/fine-tuning.

        For many pre-trained sentiment models, explicit training on custom small
        datasets might not be immediately necessary for good performance (zero-shot).
        Actual fine-tuning would require a more complex setup (e.g., using
        Hugging Face Trainer API).

        Args:
            X: Input features (not used in this basic zero-shot version).
            y: Target labels (not used in this basic zero-shot version).
            **kwargs: Additional training arguments.
        """
        if self.sentiment_pipeline:
            print(f"The model '{self.model_name}' is a pre-trained model. "
                  "Fine-tuning is not implemented in this basic version. "
                  "Using it for zero-shot predictions.")
        else:
            print("Cannot 'train' as the transformer model failed to load.")


    def predict(self, X):
        """Predicts sentiment for a list of text samples.

        Args:
            X (list[str]): A list of text samples.

        Returns:
            list[str]: A list of predicted sentiment labels (e.g., 'POSITIVE', 'NEGATIVE').
                       Returns an empty list if the model isn't loaded or input is empty.
        """
        if not self.sentiment_pipeline:
            print("Sentiment pipeline not available. Cannot predict.")
            return []
        if not X:
            return []

        try:
            results = self.sentiment_pipeline(X)
            # The pipeline returns a list of dictionaries, e.g., [{'label': 'POSITIVE', 'score': 0.99}]
            # We need to extract the 'label' and map it to our desired output format
            # if necessary. 'distilbert-base-uncased-finetuned-sst-2-english'
            # already outputs 'POSITIVE' or 'NEGATIVE'.
            # Other models might output 'LABEL_0', 'LABEL_1', etc.
            # This example assumes direct compatibility for simplicity with sst-2-english.
            predictions = [result['label'].lower() for result in results]
            return predictions
        except Exception as e:
            print(f"Error during prediction with transformer model: {e}")
            return ["error"] * len(X) # Return error placeholders


if __name__ == '__main__':
    # Example Usage:
    print("Testing TransformerSentimentModel...")
    transformer_model = TransformerSentimentModel()

    if transformer_model.sentiment_pipeline:
        sample_texts_transformer = [
            "This is a fantastic movie, I loved every minute of it!",
            "What a complete waste of time, the plot was nonsensical.",
            "It was an okay experience, not great but not terrible either."
        ]
        print(f"\nInput texts: {sample_texts_transformer}")

        # Test train method (placeholder)
        transformer_model.train([], [])

        # Test predict method
        predictions = transformer_model.predict(sample_texts_transformer)
        print(f"Predictions: {predictions}")

        # Example of how the output looks for 'distilbert-base-uncased-finetuned-sst-2-english'
        # Predictions: [{'label': 'POSITIVE', 'score': 0.99...}, {'label': 'NEGATIVE', 'score': 0.99...}, {'label': 'POSITIVE', 'score': 0.9...}]
        # We need to map this to 'positive' or 'negative' if our convention is lowercase
        # The current predict method already converts to lowercase.

        # To match SentimentModel's output (e.g. 'positive', 'negative')
        # the current implementation of predict() already handles this.
        # For 'distilbert-base-uncased-finetuned-sst-2-english'
        # labels are 'POSITIVE' or 'NEGATIVE'.
        # The .lower() call in predict() handles this.

        # If a different model returns, e.g. 'LABEL_1' for positive and 'LABEL_0' for negative
        # a mapping would be needed:
        # label_map = {'LABEL_1': 'positive', 'LABEL_0': 'negative'}
        # predictions = [label_map.get(result['label'], 'unknown') for result in results]

    else:
        print("Transformer model could not be initialized. Skipping tests.")
