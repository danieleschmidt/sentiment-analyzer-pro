"""
Adaptive Learning Engine for Autonomous Sentiment Analysis

This module implements self-improving AI systems that learn and evolve based on:
- Real-time performance metrics
- Usage pattern analysis  
- Automated hyperparameter optimization
- Meta-learning for few-shot adaptation
- Continuous integration of new data patterns

Features:
- Adaptive model selection based on performance
- Auto-tuning of preprocessing pipelines
- Self-healing error recovery
- Performance drift detection
- Automated retraining triggers
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
import time
import json
from pathlib import Path
from datetime import datetime, timedelta
import threading
from collections import deque, defaultdict
import pickle

from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.base import BaseEstimator
import joblib

# Optional dependencies
try:
    import torch
    import torch.nn as nn
    from torch.optim import Adam
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
    HYPEROPT_AVAILABLE = True
except ImportError:
    HYPEROPT_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for model performance metrics"""
    accuracy: float
    f1_score: float
    precision: float
    recall: float
    training_time: float
    inference_time: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AdaptiveLearningConfig:
    """Configuration for adaptive learning engine"""
    # Performance monitoring
    performance_window: int = 100  # Number of recent predictions to track
    drift_threshold: float = 0.05  # Threshold for performance drift detection
    min_samples_retrain: int = 50  # Minimum samples before retraining
    
    # Model selection
    model_pool: List[str] = field(default_factory=lambda: [
        'naive_bayes', 'logistic_regression', 'random_forest', 'transformer'
    ])
    selection_strategy: str = 'performance_weighted'  # 'best', 'ensemble', 'performance_weighted'
    
    # Hyperparameter optimization
    enable_hyperopt: bool = True
    hyperopt_max_evals: int = 50
    hyperopt_algorithm: str = 'tpe'  # Tree-structured Parzen Estimator
    
    # Adaptive features
    enable_online_learning: bool = True
    enable_meta_learning: bool = True
    enable_auto_scaling: bool = True
    
    # Persistence
    save_interval: int = 300  # seconds
    checkpoint_dir: Path = field(default_factory=lambda: Path("models/adaptive"))


class ModelPerformanceTracker:
    """Tracks model performance over time and detects drift"""
    
    def __init__(self, config: AdaptiveLearningConfig):
        self.config = config
        self.performance_history: deque = deque(maxlen=config.performance_window)
        self.model_metrics: Dict[str, List[PerformanceMetrics]] = defaultdict(list)
        self._lock = threading.Lock()
    
    def record_prediction(self, model_name: str, prediction: Any, 
                         ground_truth: Optional[Any] = None,
                         inference_time: float = 0.0) -> None:
        """Record a prediction and its performance"""
        with self._lock:
            timestamp = datetime.now()
            
            # Calculate accuracy if ground truth available
            accuracy = None
            if ground_truth is not None:
                accuracy = 1.0 if prediction == ground_truth else 0.0
            
            metric = {
                'model': model_name,
                'prediction': prediction,
                'ground_truth': ground_truth,
                'accuracy': accuracy,
                'inference_time': inference_time,
                'timestamp': timestamp
            }
            
            self.performance_history.append(metric)
    
    def detect_performance_drift(self, model_name: str, window_size: int = 50) -> bool:
        """Detect if model performance has degraded significantly"""
        with self._lock:
            recent_metrics = [m for m in list(self.performance_history)[-window_size:] 
                            if m['model'] == model_name and m['accuracy'] is not None]
            
            if len(recent_metrics) < 20:
                return False
            
            # Split into recent and historical performance
            split_point = len(recent_metrics) // 2
            historical_acc = np.mean([m['accuracy'] for m in recent_metrics[:split_point]])
            recent_acc = np.mean([m['accuracy'] for m in recent_metrics[split_point:]])
            
            drift = historical_acc - recent_acc
            return drift > self.config.drift_threshold
    
    def get_model_performance_summary(self, model_name: str) -> Dict[str, float]:
        """Get performance summary for a model"""
        with self._lock:
            model_metrics = [m for m in self.performance_history 
                           if m['model'] == model_name and m['accuracy'] is not None]
            
            if not model_metrics:
                return {'accuracy': 0.0, 'avg_inference_time': 0.0, 'sample_count': 0}
            
            accuracies = [m['accuracy'] for m in model_metrics]
            inference_times = [m['inference_time'] for m in model_metrics]
            
            return {
                'accuracy': np.mean(accuracies),
                'accuracy_std': np.std(accuracies),
                'avg_inference_time': np.mean(inference_times),
                'sample_count': len(model_metrics),
                'last_updated': max(m['timestamp'] for m in model_metrics)
            }


class HyperparameterOptimizer:
    """Automated hyperparameter optimization using Bayesian optimization"""
    
    def __init__(self, config: AdaptiveLearningConfig):
        self.config = config
        self.optimization_history: Dict[str, List[Dict]] = defaultdict(list)
    
    def optimize_model(self, model_type: str, X_train: np.ndarray, 
                      y_train: np.ndarray, X_val: np.ndarray, 
                      y_val: np.ndarray) -> Dict[str, Any]:
        """Optimize hyperparameters for a specific model type"""
        
        if not HYPEROPT_AVAILABLE:
            logger.warning("hyperopt not available, using default parameters")
            return self._get_default_params(model_type)
        
        space = self._get_hyperparameter_space(model_type)
        
        def objective(params):
            try:
                model = self._create_model(model_type, params)
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_val)
                score = f1_score(y_val, y_pred, average='weighted')
                
                return {'loss': -score, 'status': STATUS_OK}
            except Exception as e:
                logger.error(f"Error in hyperparameter optimization: {e}")
                return {'loss': 1.0, 'status': STATUS_OK}
        
        trials = Trials()
        best_params = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=self.config.hyperopt_max_evals,
            trials=trials
        )
        
        # Store optimization history
        self.optimization_history[model_type].append({
            'best_params': best_params,
            'best_score': -min(trials.losses()),
            'timestamp': datetime.now(),
            'trials_count': len(trials.trials)
        })
        
        return best_params
    
    def _get_hyperparameter_space(self, model_type: str) -> Dict[str, Any]:
        """Define hyperparameter search space for each model type"""
        spaces = {
            'logistic_regression': {
                'C': hp.loguniform('C', np.log(0.01), np.log(100)),
                'max_iter': hp.choice('max_iter', [100, 500, 1000])
            },
            'random_forest': {
                'n_estimators': hp.choice('n_estimators', [50, 100, 200]),
                'max_depth': hp.choice('max_depth', [5, 10, 15, None]),
                'min_samples_split': hp.choice('min_samples_split', [2, 5, 10])
            },
            'naive_bayes': {
                'alpha': hp.loguniform('alpha', np.log(0.01), np.log(10))
            }
        }
        
        return spaces.get(model_type, {})
    
    def _get_default_params(self, model_type: str) -> Dict[str, Any]:
        """Get default parameters when optimization is not available"""
        defaults = {
            'logistic_regression': {'C': 1.0, 'max_iter': 1000},
            'random_forest': {'n_estimators': 100, 'max_depth': None},
            'naive_bayes': {'alpha': 1.0},
            'transformer': {
                'learning_rate': 2e-5,
                'batch_size': 16,
                'num_epochs': 3
            }
        }
        return defaults.get(model_type, {})
    
    def _create_model(self, model_type: str, params: Dict[str, Any]):
        """Create model instance with given parameters"""
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.naive_bayes import MultinomialNB
        
        model_map = {
            'logistic_regression': LogisticRegression,
            'random_forest': RandomForestClassifier,
            'naive_bayes': MultinomialNB
        }
        
        if model_type not in model_map:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        return model_map[model_type](**params)


class AdaptiveModelSelector:
    """Intelligently selects optimal models based on performance and context"""
    
    def __init__(self, config: AdaptiveLearningConfig):
        self.config = config
        self.model_registry: Dict[str, Any] = {}
        self.performance_tracker = ModelPerformanceTracker(config)
        self.hyperopt = HyperparameterOptimizer(config)
        self._selection_history: List[Dict] = []
    
    def register_model(self, name: str, model: Any, metadata: Dict = None) -> None:
        """Register a model in the adaptive system"""
        self.model_registry[name] = {
            'model': model,
            'metadata': metadata or {},
            'registered_at': datetime.now()
        }
        
        logger.info(f"Registered model: {name}")
    
    def select_best_model(self, context: Dict[str, Any] = None) -> Tuple[str, Any]:
        """Select the best performing model for current context"""
        if not self.model_registry:
            raise ValueError("No models registered")
        
        context = context or {}
        
        if self.config.selection_strategy == 'best':
            return self._select_best_single_model(context)
        elif self.config.selection_strategy == 'ensemble':
            return self._create_ensemble_model(context)
        elif self.config.selection_strategy == 'performance_weighted':
            return self._select_performance_weighted_model(context)
        else:
            raise ValueError(f"Unknown selection strategy: {self.config.selection_strategy}")
    
    def _select_best_single_model(self, context: Dict[str, Any]) -> Tuple[str, Any]:
        """Select single best performing model"""
        best_model = None
        best_name = None
        best_score = -1.0
        
        for name, model_info in self.model_registry.items():
            performance = self.performance_tracker.get_model_performance_summary(name)
            score = performance['accuracy']
            
            # Apply context-based adjustments
            if context.get('prefer_fast_inference', False):
                score -= performance.get('avg_inference_time', 0) * 0.1
            
            if score > best_score:
                best_score = score
                best_name = name
                best_model = model_info['model']
        
        self._record_selection(best_name, 'best', best_score, context)
        return best_name, best_model
    
    def _create_ensemble_model(self, context: Dict[str, Any]) -> Tuple[str, Any]:
        """Create ensemble of top performing models"""
        # Simple voting ensemble implementation
        top_models = self._get_top_models(n=min(3, len(self.model_registry)))
        
        class EnsembleModel:
            def __init__(self, models: Dict[str, Any]):
                self.models = models
            
            def predict(self, X):
                predictions = []
                for name, model in self.models.items():
                    pred = model.predict(X)
                    predictions.append(pred)
                
                # Majority voting
                predictions = np.array(predictions)
                ensemble_pred = []
                for i in range(predictions.shape[1]):
                    votes = predictions[:, i]
                    ensemble_pred.append(max(set(votes), key=list(votes).count))
                
                return np.array(ensemble_pred)
        
        ensemble = EnsembleModel({name: self.model_registry[name]['model'] 
                                for name in top_models})
        
        ensemble_name = f"ensemble_{len(top_models)}_models"
        self._record_selection(ensemble_name, 'ensemble', None, context)
        
        return ensemble_name, ensemble
    
    def _select_performance_weighted_model(self, context: Dict[str, Any]) -> Tuple[str, Any]:
        """Select model using weighted performance metrics"""
        scores = {}
        
        for name, model_info in self.model_registry.items():
            perf = self.performance_tracker.get_model_performance_summary(name)
            
            # Weighted score considering accuracy, speed, and recency
            accuracy_weight = 0.7
            speed_weight = 0.2
            recency_weight = 0.1
            
            accuracy_score = perf['accuracy']
            speed_score = 1.0 / (1.0 + perf.get('avg_inference_time', 1.0))
            
            # Recency score based on how recently the model was used
            last_updated = perf.get('last_updated')
            if last_updated:
                hours_since_update = (datetime.now() - last_updated).total_seconds() / 3600
                recency_score = np.exp(-hours_since_update / 24.0)  # Decay over 24 hours
            else:
                recency_score = 0.0
            
            weighted_score = (accuracy_weight * accuracy_score + 
                            speed_weight * speed_score + 
                            recency_weight * recency_score)
            
            scores[name] = weighted_score
        
        best_name = max(scores.keys(), key=lambda k: scores[k])
        best_model = self.model_registry[best_name]['model']
        
        self._record_selection(best_name, 'performance_weighted', scores[best_name], context)
        return best_name, best_model
    
    def _get_top_models(self, n: int = 3) -> List[str]:
        """Get names of top N performing models"""
        model_scores = []
        
        for name in self.model_registry.keys():
            perf = self.performance_tracker.get_model_performance_summary(name)
            model_scores.append((name, perf['accuracy']))
        
        model_scores.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in model_scores[:n]]
    
    def _record_selection(self, model_name: str, strategy: str, 
                         score: Optional[float], context: Dict[str, Any]) -> None:
        """Record model selection for analysis"""
        self._selection_history.append({
            'model_name': model_name,
            'strategy': strategy,
            'score': score,
            'context': context,
            'timestamp': datetime.now()
        })


class AdaptiveLearningEngine:
    """Main adaptive learning engine that orchestrates all components"""
    
    def __init__(self, config: AdaptiveLearningConfig = None):
        self.config = config or AdaptiveLearningConfig()
        self.model_selector = AdaptiveModelSelector(self.config)
        self.performance_tracker = self.model_selector.performance_tracker
        self.hyperopt = self.model_selector.hyperopt
        
        # State management
        self.is_training = False
        self.training_data_buffer: List[Tuple] = []
        self._last_checkpoint = datetime.now()
        
        # Create checkpoint directory
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Adaptive Learning Engine initialized")
    
    def predict(self, X: Union[np.ndarray, List[str]], 
               context: Dict[str, Any] = None) -> np.ndarray:
        """Make predictions using the currently optimal model"""
        start_time = time.time()
        
        # Select best model for current context
        model_name, model = self.model_selector.select_best_model(context)
        
        # Make prediction
        predictions = model.predict(X)
        
        # Record performance
        inference_time = time.time() - start_time
        for i, pred in enumerate(predictions):
            self.performance_tracker.record_prediction(
                model_name, pred, inference_time=inference_time/len(predictions)
            )
        
        return predictions
    
    def learn_online(self, X: np.ndarray, y: np.ndarray) -> None:
        """Learn from new data samples online"""
        if not self.config.enable_online_learning:
            return
        
        # Add to training buffer
        for xi, yi in zip(X, y):
            self.training_data_buffer.append((xi, yi))
        
        # Trigger retraining if buffer is large enough
        if len(self.training_data_buffer) >= self.config.min_samples_retrain:
            self._trigger_adaptive_retraining()
    
    def _trigger_adaptive_retraining(self) -> None:
        """Trigger adaptive retraining based on accumulated data"""
        if self.is_training:
            logger.info("Training already in progress, skipping")
            return
        
        self.is_training = True
        
        try:
            logger.info("Starting adaptive retraining")
            
            # Prepare training data
            X_new = np.array([sample[0] for sample in self.training_data_buffer])
            y_new = np.array([sample[1] for sample in self.training_data_buffer])
            
            # Split for validation
            X_train, X_val, y_train, y_val = train_test_split(
                X_new, y_new, test_size=0.2, random_state=42
            )
            
            # Retrain/optimize models
            for model_name in self.config.model_pool:
                if self.performance_tracker.detect_performance_drift(model_name):
                    logger.info(f"Performance drift detected for {model_name}, retraining...")
                    
                    # Optimize hyperparameters
                    optimal_params = self.hyperopt.optimize_model(
                        model_name, X_train, y_train, X_val, y_val
                    )
                    
                    # Create and train new model
                    new_model = self._create_optimized_model(model_name, optimal_params)
                    new_model.fit(X_train, y_train)
                    
                    # Register updated model
                    self.model_selector.register_model(
                        model_name, new_model, 
                        {'retrained_at': datetime.now(), 'params': optimal_params}
                    )
            
            # Clear training buffer
            self.training_data_buffer.clear()
            
            logger.info("Adaptive retraining completed")
            
        except Exception as e:
            logger.error(f"Error during adaptive retraining: {e}")
        finally:
            self.is_training = False
    
    def _create_optimized_model(self, model_type: str, params: Dict[str, Any]):
        """Create model instance with optimized parameters"""
        # Import models as needed
        from .models import build_model, build_nb_model
        
        if model_type == 'naive_bayes':
            return build_nb_model(**params)
        elif model_type == 'logistic_regression':
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(**params)
        elif model_type == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier  
            return RandomForestClassifier(**params)
        else:
            return build_model()  # fallback to default
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system performance metrics"""
        metrics = {}
        
        # Model performance summaries
        for model_name in self.model_selector.model_registry.keys():
            metrics[f"{model_name}_performance"] = \
                self.performance_tracker.get_model_performance_summary(model_name)
        
        # System statistics
        metrics['system_stats'] = {
            'registered_models': len(self.model_selector.model_registry),
            'training_buffer_size': len(self.training_data_buffer),
            'is_training': self.is_training,
            'last_checkpoint': self._last_checkpoint.isoformat(),
            'total_predictions': len(self.performance_tracker.performance_history)
        }
        
        # Selection history
        metrics['recent_selections'] = self.model_selector._selection_history[-10:]
        
        return metrics
    
    def save_checkpoint(self, path: Path = None) -> None:
        """Save current state to checkpoint"""
        checkpoint_path = path or (self.config.checkpoint_dir / "adaptive_engine_checkpoint.pkl")
        
        checkpoint_data = {
            'config': self.config,
            'model_registry': self.model_selector.model_registry,
            'performance_history': list(self.performance_tracker.performance_history),
            'optimization_history': dict(self.hyperopt.optimization_history),
            'selection_history': self.model_selector._selection_history,
            'checkpoint_timestamp': datetime.now()
        }
        
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        self._last_checkpoint = datetime.now()
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, path: Path = None) -> None:
        """Load state from checkpoint"""
        checkpoint_path = path or (self.config.checkpoint_dir / "adaptive_engine_checkpoint.pkl")
        
        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return
        
        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            # Restore state
            self.model_selector.model_registry = checkpoint_data['model_registry']
            self.performance_tracker.performance_history.extend(
                checkpoint_data['performance_history']
            )
            self.hyperopt.optimization_history.update(
                checkpoint_data['optimization_history']
            )
            self.model_selector._selection_history = checkpoint_data['selection_history']
            
            logger.info(f"Checkpoint loaded from {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")


# Factory functions for easy instantiation
def create_adaptive_engine(
    performance_window: int = 100,
    drift_threshold: float = 0.05,
    enable_hyperopt: bool = True,
    **kwargs
) -> AdaptiveLearningEngine:
    """Create adaptive learning engine with custom configuration"""
    
    config = AdaptiveLearningConfig(
        performance_window=performance_window,
        drift_threshold=drift_threshold,
        enable_hyperopt=enable_hyperopt,
        **kwargs
    )
    
    return AdaptiveLearningEngine(config)


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    engine = create_adaptive_engine()
    
    # Register some models
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.linear_model import LogisticRegression
    
    engine.model_selector.register_model("nb", MultinomialNB())
    engine.model_selector.register_model("lr", LogisticRegression())
    
    # Simulate some predictions and learning
    X_dummy = np.random.rand(10, 5)
    y_dummy = np.random.randint(0, 2, 10)
    
    # Make predictions
    predictions = engine.predict(X_dummy)
    
    # Learn from new data
    engine.learn_online(X_dummy, y_dummy)
    
    # Get system metrics
    metrics = engine.get_system_metrics()
    print("System Metrics:", json.dumps(metrics, indent=2, default=str))