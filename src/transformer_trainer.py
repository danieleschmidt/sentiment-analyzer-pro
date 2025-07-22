"""BERT fine-tuning pipeline for sentiment analysis."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import os

try:
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
except Exception:  # pragma: no cover - optional dependency
    pd = None
    np = None
    train_test_split = None
    accuracy_score = None
    precision_recall_fscore_support = None
    confusion_matrix = None

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import (
        DistilBertTokenizer,
        DistilBertForSequenceClassification,
        AdamW,
        get_linear_schedule_with_warmup,
        Trainer,
        TrainingArguments,
        EarlyStoppingCallback,
    )
except Exception:  # pragma: no cover - optional dependency
    torch = None
    Dataset = None
    DataLoader = None
    DistilBertTokenizer = None
    DistilBertForSequenceClassification = None
    AdamW = None
    get_linear_schedule_with_warmup = None
    Trainer = None
    TrainingArguments = None
    EarlyStoppingCallback = None

from .preprocessing import clean_text

logger = logging.getLogger(__name__)


@dataclass
class TransformerConfig:
    """Configuration for transformer training."""
    model_name: str = "distilbert-base-uncased"
    max_length: int = 128
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    output_dir: str = "models/transformer"
    save_strategy: str = "epoch"
    evaluation_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_accuracy"
    greater_is_better: bool = True
    early_stopping_patience: int = 2


if Dataset is not None:
    class SentimentDataset(Dataset):
        """Dataset class for sentiment analysis."""
        
        def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_length = max_length
        
        def __len__(self) -> int:
            return len(self.texts)
        
        def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
            text = str(self.texts[idx])
            label = self.labels[idx]
            
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long)
            }
else:
    class SentimentDataset:
        """Placeholder class when torch is not available."""
        
        def __init__(self, *args, **kwargs):
            raise ImportError("torch is required for SentimentDataset")


class TransformerTrainer:
    """BERT fine-tuning trainer for sentiment analysis."""
    
    def __init__(self, config: Optional[TransformerConfig] = None):
        if torch is None or DistilBertTokenizer is None:
            raise ImportError("torch and transformers are required for TransformerTrainer")
        
        self.config = config or TransformerConfig()
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.label_map = {}
        self.reverse_label_map = {}
        
    def _prepare_labels(self, labels: pd.Series) -> Tuple[List[int], Dict[str, int]]:
        """Convert string labels to integers."""
        unique_labels = sorted(labels.unique())
        label_map = {label: idx for idx, label in enumerate(unique_labels)}
        reverse_label_map = {idx: label for label, idx in label_map.items()}
        
        numeric_labels = [label_map[label] for label in labels]
        
        self.label_map = label_map
        self.reverse_label_map = reverse_label_map
        
        logger.info(f"Label mapping: {label_map}")
        return numeric_labels, label_map
    
    def _setup_model(self, num_labels: int):
        """Initialize tokenizer and model."""
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.config.model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=num_labels
        )
        
        logger.info(f"Initialized {self.config.model_name} with {num_labels} labels")
    
    def _compute_metrics(self, eval_pred):
        """Compute evaluation metrics."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def train(
        self,
        csv_path: str,
        test_size: float = 0.2,
        validation_size: float = 0.1,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """Train the transformer model on sentiment data."""
        if pd is None or train_test_split is None:
            raise ImportError("pandas and scikit-learn are required for training")
        
        # Load and preprocess data
        logger.info(f"Loading data from {csv_path}")
        data = pd.read_csv(csv_path)
        
        if 'text' not in data.columns or 'label' not in data.columns:
            raise ValueError("Data must contain 'text' and 'label' columns")
        
        # Clean text data
        texts = data['text'].apply(clean_text).tolist()
        labels, label_map = self._prepare_labels(data['label'])
        
        # Setup model
        num_labels = len(label_map)
        self._setup_model(num_labels)
        
        # Split data
        train_texts, temp_texts, train_labels, temp_labels = train_test_split(
            texts, labels, test_size=(test_size + validation_size), random_state=random_state
        )
        
        if validation_size > 0:
            val_size = validation_size / (test_size + validation_size)
            val_texts, test_texts, val_labels, test_labels = train_test_split(
                temp_texts, temp_labels, test_size=val_size, random_state=random_state
            )
        else:
            val_texts, val_labels = temp_texts, temp_labels
            test_texts, test_labels = [], []
        
        # Create datasets
        train_dataset = SentimentDataset(
            train_texts, train_labels, self.tokenizer, self.config.max_length
        )
        val_dataset = SentimentDataset(
            val_texts, val_labels, self.tokenizer, self.config.max_length
        )
        
        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            logging_dir=f"{self.config.output_dir}/logs",
            logging_steps=10,
            evaluation_strategy=self.config.evaluation_strategy,
            save_strategy=self.config.save_strategy,
            load_best_model_at_end=self.config.load_best_model_at_end,
            metric_for_best_model=self.config.metric_for_best_model,
            greater_is_better=self.config.greater_is_better,
            learning_rate=self.config.learning_rate,
        )
        
        # Setup trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self._compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=self.config.early_stopping_patience)]
        )
        
        # Train model
        logger.info("Starting training...")
        train_result = self.trainer.train()
        
        # Evaluate on validation set
        logger.info("Evaluating on validation set...")
        eval_result = self.trainer.evaluate()
        
        # Test set evaluation if available
        test_result = {}
        if test_texts:
            test_dataset = SentimentDataset(
                test_texts, test_labels, self.tokenizer, self.config.max_length
            )
            logger.info("Evaluating on test set...")
            test_result = self.trainer.evaluate(eval_dataset=test_dataset)
            test_result = {f"test_{k}": v for k, v in test_result.items()}
        
        # Save model and tokenizer
        self.save_model()
        
        results = {
            'train_loss': train_result.training_loss,
            'eval_accuracy': eval_result['eval_accuracy'],
            'eval_f1': eval_result['eval_f1'],
            'eval_precision': eval_result['eval_precision'],
            'eval_recall': eval_result['eval_recall'],
            'label_map': label_map,
            'num_samples': {
                'train': len(train_texts),
                'validation': len(val_texts),
                'test': len(test_texts) if test_texts else 0
            }
        }
        results.update(test_result)
        
        logger.info(f"Training completed. Results: {results}")
        return results
    
    def predict(self, texts: List[str]) -> List[str]:
        """Make predictions on new texts."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not trained. Call train() first or load a trained model.")
        
        predictions = []
        for text in texts:
            inputs = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.config.max_length,
                return_tensors='pt'
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                predicted_class_id = outputs.logits.argmax().item()
                predicted_label = self.reverse_label_map[predicted_class_id]
                predictions.append(predicted_label)
        
        return predictions
    
    def save_model(self, path: Optional[str] = None) -> str:
        """Save the trained model and tokenizer."""
        save_path = path or self.config.output_dir
        os.makedirs(save_path, exist_ok=True)
        
        if self.trainer:
            self.trainer.save_model(save_path)
        else:
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
        
        # Save label mappings
        import json
        with open(f"{save_path}/label_map.json", 'w') as f:
            json.dump(self.label_map, f)
        
        logger.info(f"Model saved to {save_path}")
        return save_path
    
    def load_model(self, path: str):
        """Load a trained model and tokenizer."""
        import json
        
        self.tokenizer = DistilBertTokenizer.from_pretrained(path)
        self.model = DistilBertForSequenceClassification.from_pretrained(path)
        
        # Load label mappings
        label_map_path = f"{path}/label_map.json"
        if os.path.exists(label_map_path):
            with open(label_map_path, 'r') as f:
                self.label_map = json.load(f)
                self.reverse_label_map = {v: k for k, v in self.label_map.items()}
        
        logger.info(f"Model loaded from {path}")


def train_transformer_model(
    csv_path: str = "data/sample_reviews.csv",
    config: Optional[TransformerConfig] = None
) -> Dict[str, Any]:
    """Convenience function to train a transformer model."""
    trainer = TransformerTrainer(config)
    return trainer.train(csv_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    config = TransformerConfig(
        num_epochs=2,  # Reduced for testing
        batch_size=8,  # Smaller batch size for testing
        output_dir="models/distilbert_sentiment"
    )
    
    results = train_transformer_model(config=config)
    print(f"Training Results: {results}")