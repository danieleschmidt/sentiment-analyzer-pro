from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import numpy as np # For array manipulation if needed

class TransformerSentimentModel:
    """A sentiment analysis model using Hugging Face Transformers.
    This model uses a pre-trained transformer model for sentiment analysis.
    It can perform zero-shot prediction. Fine-tuning is not implemented
    in this basic version but can be added.

    Attributes:
        model_name (str): The name of the pre-trained Hugging Face model.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer for the model.
        model (transformers.PreTrainedModel): The loaded transformer model.
        device (torch.device): The device (CPU or CUDA) the model is on.
        id2label (dict): Mapping from class ID to label name from model config.
        label2id (dict): Mapping from label name to class ID from model config.
    """

    def __init__(self, model_name='distilbert-base-uncased-finetuned-sst-2-english'):
        """Initializes the TransformerSentimentModel.

        Args:
            model_name (str, optional): Name of the pre-trained Hugging Face model.
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device} for Transformer model.")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            print(f"Transformer model '{self.model_name}' and tokenizer loaded successfully.")
            # Storing label mapping if available in model config
            self.id2label = self.model.config.id2label
            self.label2id = self.model.config.label2id # Added label2id
            print(f"Model label mapping (id2label): {self.id2label}")
            print(f"Model label mapping (label2id): {self.label2id}")


        except Exception as e:
            print(f"Error loading transformer model '{self.model_name}': {e}")
            self.tokenizer = None
            self.model = None
            self.id2label = None
            self.label2id = None # Ensure label2id is also None on error
    
    def train(self, X, y, **kwargs):
        """Placeholder for training/fine-tuning.
        For this version, it just ensures the model is loaded.
        """
        if self.model and self.tokenizer:
            print(f"The model '{self.model_name}' is a pre-trained model. "
                  "Fine-tuning is not implemented in this version. Using for zero-shot predictions.")
        else:
            print("Cannot 'train' as the transformer model failed to load.")

    def predict(self, texts: list, return_probabilities=False):
        """Predicts sentiment for a list of text samples.

        Args:
            texts (list[str]): A list of text samples.
            return_probabilities (bool, optional): If True, returns probabilities
                                                  for the positive class alongside labels.
                                                  Defaults to False.

        Returns:
            list[str] or tuple(list[str], list[float]):
                - A list of predicted sentiment labels (e.g., 'positive', 'negative').
                - If return_probabilities is True, a tuple containing:
                    - list[str]: predicted labels.
                    - list[float]: probabilities for the positive class.
                       (Assumes binary classification positive/negative for this score)
        """
        if not self.model or not self.tokenizer:
            print("Transformer model or tokenizer not available. Cannot predict.")
            # Ensure return type matches expected output structure even on early exit
            return ([], []) if return_probabilities else []
        if not texts: # Handle empty input list
            return ([], []) if return_probabilities else []

        try:
            valid_texts = [str(text) if text is not None else "" for text in texts]
            inputs = self.tokenizer(
                valid_texts, padding=True, truncation=True,
                max_length=self.tokenizer.model_max_length if hasattr(self.tokenizer, 'model_max_length') and self.tokenizer.model_max_length is not None else 512,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

            probabilities_all_classes = F.softmax(logits, dim=-1)
            predicted_class_ids = torch.argmax(probabilities_all_classes, dim=-1).cpu().tolist()

            if self.id2label:
                predictions = [self.id2label[class_id].lower() for class_id in predicted_class_ids]
            else: # Fallback if id2label is not available
                print("Warning: id2label mapping not found in model config. Returning raw class IDs as strings.")
                predictions = [str(class_id) for class_id in predicted_class_ids]

            if return_probabilities:
                positive_class_id = None
                if self.label2id: # Primary way to find 'positive' class ID
                    for label, id_val in self.label2id.items():
                        if label.lower() == 'positive':
                            positive_class_id = id_val
                            break
                
                if positive_class_id is None and self.id2label: # Fallback using id2label
                    for id_val, label_str in self.id2label.items():
                        if label_str.lower() == 'positive':
                            positive_class_id = id_val
                            break
                
                if positive_class_id is None: # Further fallback for binary classification
                    if probabilities_all_classes.shape[1] == 2:
                        print("Warning: Could not definitively determine 'positive' class ID. Assuming class ID 1 (the higher index) is 'positive' for binary classification.")
                        positive_class_id = 1 # Common for SST-2 like models where 0 is neg, 1 is pos
                    else:
                        print("Error: Cannot determine positive class ID for probabilities. Multiclass or unknown mapping. Returning 0.0 for all positive probabilities.")
                        positive_class_probs = [0.0] * len(texts)
                        return predictions, positive_class_probs
                
                # Ensure positive_class_id is within bounds of the probability tensor
                if positive_class_id >= probabilities_all_classes.shape[1]:
                    print(f"Error: Determined positive_class_id {positive_class_id} is out of bounds for probability tensor shape {probabilities_all_classes.shape}. Returning 0.0 for all probabilities.")
                    positive_class_probs = [0.0] * len(texts)
                else:
                    positive_class_probs = probabilities_all_classes[:, positive_class_id].cpu().tolist()
                
                return predictions, positive_class_probs
            else:
                return predictions

        except Exception as e:
            print(f"Error during transformer prediction: {e}")
            # Prepare error values matching the expected return type
            error_val_labels = ["error"] * len(texts)
            if return_probabilities:
                error_val_probs = [0.0] * len(texts) # Default probability for error cases
                return error_val_labels, error_val_probs
            else:
                return error_val_labels

if __name__ == '__main__':
    print("Testing TransformerSentimentModel with probabilities...")
    # This model will download files on first run if not cached by Hugging Face.
    transformer_model = TransformerSentimentModel() # Uses default distilbert model

    if transformer_model.model: # Check if model loaded successfully
        sample_texts = [
            "This is a fantastic movie, I loved every minute of it!",
            "What a complete waste of time, the plot was nonsensical.",
            "It was an okay experience, not great but not terrible either.",
            "The product is amazing, works as expected.",
            "I'm very disappointed with the quality.",
            None, 
            ""    
        ]
        print(f"\nInput texts for Transformer: {sample_texts}")
        
        # Test labels only
        labels_only = transformer_model.predict(sample_texts, return_probabilities=False)
        print(f"\nTransformer Predictions (labels only):")
        for text, pred_label in zip(sample_texts, labels_only):
            text_display = str(text)[:30] if text is not None else "None"
            print(f"  Text: \"{text_display}...\" => Prediction: {pred_label}")

        # Test labels and probabilities
        labels_and_probs_output = transformer_model.predict(sample_texts, return_probabilities=True)
        print(f"\nTransformer Predictions (labels and positive class probabilities):")
        if labels_and_probs_output and len(labels_and_probs_output) == 2:
            preds, probs = labels_and_probs_output
            for text, p_label, prob_score in zip(sample_texts, preds, probs):
                text_display = str(text)[:30] if text is not None else "None"
                print(f"  Text: \"{text_display}...\" => Prediction: {p_label}, Positive Prob: {prob_score:.4f}")
        else:
            print("Error: predict method did not return the expected tuple for labels and probabilities.")
            
    else:
        print("Transformer model could not be initialized. Skipping example usage.")
