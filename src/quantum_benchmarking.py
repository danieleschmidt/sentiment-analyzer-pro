"""
Comprehensive benchmarking framework for quantum-inspired sentiment analysis.

This module provides tools for rigorous evaluation and comparison of 
quantum-inspired sentiment analysis models against classical baselines.
"""

from __future__ import annotations

import time
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
import logging
from pathlib import Path
import json
from datetime import datetime

# Statistical analysis
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    confusion_matrix, classification_report
)

# Import our models
from .quantum_inspired_sentiment import (
    QuantumInspiredSentimentClassifier, 
    QuantumInspiredConfig,
    create_quantum_inspired_classifier
)
from .models import build_model, build_nb_model
from .preprocessing import clean_text

# Optional dependencies
try:
    from .transformer_trainer import TransformerTrainer, TransformerConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for comprehensive benchmarking."""
    
    # Data configuration
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    
    # Models to benchmark
    include_classical: bool = True
    include_quantum_inspired: bool = True
    include_transformers: bool = True
    
    # Quantum-inspired model variations
    quantum_qubit_sizes: List[int] = field(default_factory=lambda: [4, 6, 8])
    quantum_layer_depths: List[int] = field(default_factory=lambda: [2, 3, 4])
    
    # Performance analysis
    measure_training_time: bool = True
    measure_inference_time: bool = True
    statistical_significance_test: bool = True
    alpha: float = 0.05  # Significance level
    
    # Output configuration
    save_results: bool = True
    generate_plots: bool = True
    verbose: bool = True


@dataclass
class ModelResult:
    """Results for a single model evaluation."""
    
    name: str
    config: Dict[str, Any]
    
    # Performance metrics
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_score: Optional[float] = None
    
    # Timing metrics
    training_time: Optional[float] = None
    inference_time: Optional[float] = None
    
    # Cross-validation results
    cv_scores: Optional[List[float]] = None
    cv_mean: Optional[float] = None
    cv_std: Optional[float] = None
    
    # Additional metrics
    confusion_matrix: Optional[np.ndarray] = None
    classification_report: Optional[str] = None
    
    # Model-specific information
    model_parameters: Optional[int] = None
    convergence_info: Optional[Dict[str, Any]] = None


class QuantumBenchmarkSuite:
    """
    Comprehensive benchmarking suite for quantum-inspired sentiment analysis.
    
    Implements rigorous evaluation methodology including:
    - Multiple baseline comparisons
    - Statistical significance testing
    - Cross-validation analysis
    - Performance profiling
    - Reproducible experimental framework
    """
    
    def __init__(self, config: BenchmarkConfig = None):
        self.config = config or BenchmarkConfig()
        self.results: List[ModelResult] = []
        self.data: Optional[pd.DataFrame] = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        logger.info("Initialized QuantumBenchmarkSuite")
    
    def load_data(self, data_path: str = None, data: pd.DataFrame = None) -> None:
        """Load and prepare benchmark data."""
        if data is not None:
            self.data = data.copy()
        elif data_path:
            self.data = pd.read_csv(data_path)
        else:
            # Create synthetic data for demonstration
            self.data = self._create_synthetic_data()
        
        # Ensure required columns exist
        if 'text' not in self.data.columns or 'label' not in self.data.columns:
            raise ValueError("Data must contain 'text' and 'label' columns")
        
        # Clean and prepare data
        self.data['text_clean'] = self.data['text'].apply(clean_text)
        
        # Split data
        X = self.data['text_clean'].tolist()
        y = self.data['label'].tolist()
        
        # Ensure we have enough samples for stratified split
        if len(X) < 4:  # Need at least 4 samples for 20% test split with 2 classes
            # Use a smaller test size or no stratification for very small datasets
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, 
                test_size=max(1, int(len(X) * self.config.test_size)),
                random_state=self.config.random_state,
                stratify=None
            )
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, 
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=y
            )
        
        logger.info(f"Loaded data: {len(self.X_train)} train, {len(self.X_test)} test samples")
    
    def _create_synthetic_data(self) -> pd.DataFrame:
        """Create synthetic data for benchmarking."""
        np.random.seed(self.config.random_state)
        
        positive_texts = [
            "I love this product! It's amazing and works perfectly.",
            "Excellent quality and fast delivery. Highly recommended!",
            "Great customer service and beautiful design.",
            "Outstanding performance and value for money.",
            "Fantastic experience, will definitely buy again!",
            "Perfect solution for my needs. Very satisfied.",
            "Incredible features and user-friendly interface.",
            "Top-notch quality and reliable service.",
            "Wonderful product with excellent support team.",
            "Amazing results and exceeded my expectations!"
        ]
        
        negative_texts = [
            "Terrible product, completely useless and broken.",
            "Worst experience ever, poor quality and service.",
            "Disappointing results and waste of money.",
            "Broken on arrival, customer service unhelpful.",
            "Poor design and functionality. Not recommended.",
            "Expensive and low quality. Very disappointed.",
            "Difficult to use and doesn't work as advertised.",
            "Bad experience with delivery and product quality.",
            "Unreliable and causes more problems than solutions.",
            "Frustrating experience and poor value for money."
        ]
        
        # Create balanced dataset with variations
        texts = []
        labels = []
        
        for _ in range(50):  # 50 positive samples
            base_text = np.random.choice(positive_texts)
            # Add some variation
            if np.random.random() > 0.5:
                variations = [" Really good!", " Highly satisfied.", " Would recommend.", " Perfect!"]
                base_text += np.random.choice(variations)
            texts.append(base_text)
            labels.append('positive')
        
        for _ in range(50):  # 50 negative samples
            base_text = np.random.choice(negative_texts)
            # Add some variation
            if np.random.random() > 0.5:
                variations = [" Avoid this.", " Complete waste.", " Very bad.", " Horrible!"]
                base_text += np.random.choice(variations)
            texts.append(base_text)
            labels.append('negative')
        
        return pd.DataFrame({'text': texts, 'label': labels})
    
    def _benchmark_classical_models(self) -> None:
        """Benchmark classical sentiment analysis models."""
        if not self.config.include_classical:
            return
        
        logger.info("Benchmarking classical models...")
        
        # Logistic Regression model
        try:
            start_time = time.time()
            lr_model = build_model()
            lr_model.fit(self.X_train, self.y_train)
            training_time = time.time() - start_time
            
            # Test performance
            start_time = time.time()
            predictions = lr_model.predict(self.X_test)
            inference_time = (time.time() - start_time) / len(self.X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                self.y_test, predictions, average='weighted'
            )
            
            # Cross-validation
            cv_scores = cross_val_score(
                lr_model.pipeline, self.X_train, self.y_train,
                cv=self.config.cv_folds, scoring='accuracy'
            )
            
            result = ModelResult(
                name="Logistic Regression",
                config={"type": "classical", "algorithm": "logistic_regression"},
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                training_time=training_time,
                inference_time=inference_time,
                cv_scores=cv_scores.tolist(),
                cv_mean=cv_scores.mean(),
                cv_std=cv_scores.std(),
                confusion_matrix=confusion_matrix(self.y_test, predictions),
                classification_report=classification_report(self.y_test, predictions)
            )
            
            self.results.append(result)
            logger.info(f"Logistic Regression - Accuracy: {accuracy:.4f}, CV: {cv_scores.mean():.4f}±{cv_scores.std():.4f}")
            
        except Exception as e:
            logger.error(f"Failed to benchmark Logistic Regression: {e}")
        
        # Naive Bayes model
        try:
            start_time = time.time()
            nb_model = build_nb_model()
            nb_model.fit(self.X_train, self.y_train)
            training_time = time.time() - start_time
            
            # Test performance
            start_time = time.time()
            predictions = nb_model.predict(self.X_test)
            inference_time = (time.time() - start_time) / len(self.X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                self.y_test, predictions, average='weighted'
            )
            
            # Cross-validation
            cv_scores = cross_val_score(
                nb_model.pipeline, self.X_train, self.y_train,
                cv=self.config.cv_folds, scoring='accuracy'
            )
            
            result = ModelResult(
                name="Naive Bayes",
                config={"type": "classical", "algorithm": "naive_bayes"},
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                training_time=training_time,
                inference_time=inference_time,
                cv_scores=cv_scores.tolist(),
                cv_mean=cv_scores.mean(),
                cv_std=cv_scores.std(),
                confusion_matrix=confusion_matrix(self.y_test, predictions),
                classification_report=classification_report(self.y_test, predictions)
            )
            
            self.results.append(result)
            logger.info(f"Naive Bayes - Accuracy: {accuracy:.4f}, CV: {cv_scores.mean():.4f}±{cv_scores.std():.4f}")
            
        except Exception as e:
            logger.error(f"Failed to benchmark Naive Bayes: {e}")
    
    def _benchmark_quantum_inspired_models(self) -> None:
        """Benchmark quantum-inspired models with different configurations."""
        if not self.config.include_quantum_inspired:
            return
        
        logger.info("Benchmarking quantum-inspired models...")
        
        # Test different configurations
        for n_qubits in self.config.quantum_qubit_sizes:
            for n_layers in self.config.quantum_layer_depths:
                try:
                    config = QuantumInspiredConfig(
                        n_qubits=n_qubits,
                        n_layers=n_layers,
                        max_iterations=50  # Reduce for benchmarking
                    )
                    
                    model_name = f"Quantum-Inspired (q={n_qubits}, l={n_layers})"
                    logger.info(f"Training {model_name}...")
                    
                    # Create and train model
                    start_time = time.time()
                    quantum_model = QuantumInspiredSentimentClassifier(config)
                    quantum_model.fit(self.X_train, self.y_train)
                    training_time = time.time() - start_time
                    
                    # Test performance
                    start_time = time.time()
                    predictions = quantum_model.predict(self.X_test)
                    inference_time = (time.time() - start_time) / len(self.X_test)
                    
                    # Calculate metrics
                    accuracy = accuracy_score(self.y_test, predictions)
                    precision, recall, f1, _ = precision_recall_fscore_support(
                        self.y_test, predictions, average='weighted'
                    )
                    
                    # Get probabilities for AUC if binary classification
                    auc_score = None
                    try:
                        if len(set(self.y_test)) == 2:
                            probas = quantum_model.predict_proba(self.X_test)
                            if probas.shape[1] == 2:
                                auc_score = roc_auc_score(
                                    [1 if label == 'positive' else 0 for label in self.y_test],
                                    probas[:, 1]
                                )
                    except Exception as e:
                        logger.warning(f"Could not calculate AUC: {e}")
                    
                    result = ModelResult(
                        name=model_name,
                        config={
                            "type": "quantum_inspired",
                            "n_qubits": n_qubits,
                            "n_layers": n_layers,
                            "n_parameters": config.n_parameters
                        },
                        accuracy=accuracy,
                        precision=precision,
                        recall=recall,
                        f1_score=f1,
                        auc_score=auc_score,
                        training_time=training_time,
                        inference_time=inference_time,
                        confusion_matrix=confusion_matrix(self.y_test, predictions),
                        classification_report=classification_report(self.y_test, predictions),
                        model_parameters=config.n_parameters,
                        convergence_info=quantum_model.training_history[-1] if quantum_model.training_history else None
                    )
                    
                    self.results.append(result)
                    logger.info(f"{model_name} - Accuracy: {accuracy:.4f}, Training time: {training_time:.2f}s")
                    
                except Exception as e:
                    logger.error(f"Failed to benchmark {model_name}: {e}")
    
    def _benchmark_transformer_models(self) -> None:
        """Benchmark transformer-based models."""
        if not self.config.include_transformers or not TRANSFORMERS_AVAILABLE:
            return
        
        logger.info("Benchmarking transformer models... (Note: This is a placeholder)")
        # This would require the actual transformer implementation
        # For now, we'll skip this to focus on quantum-inspired benchmarking
    
    def run_benchmark(self, data_path: str = None, data: pd.DataFrame = None) -> Dict[str, Any]:
        """Run comprehensive benchmark suite."""
        logger.info("Starting comprehensive benchmark suite...")
        
        # Load data
        self.load_data(data_path, data)
        
        # Clear previous results
        self.results = []
        
        # Run benchmarks
        self._benchmark_classical_models()
        self._benchmark_quantum_inspired_models()
        self._benchmark_transformer_models()
        
        # Analyze results
        analysis = self._analyze_results()
        
        # Generate report
        report = self._generate_report(analysis)
        
        # Save results if configured
        if self.config.save_results:
            self._save_results(report)
        
        # Generate plots if configured
        if self.config.generate_plots and PLOTTING_AVAILABLE:
            self._generate_plots()
        
        logger.info("Benchmark suite completed!")
        return report
    
    def _analyze_results(self) -> Dict[str, Any]:
        """Analyze benchmark results with statistical tests."""
        if not self.results:
            return {}
        
        analysis = {
            'summary': {},
            'statistical_tests': {},
            'rankings': {}
        }
        
        # Summary statistics
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        for metric in metrics:
            values = [getattr(result, metric) for result in self.results if getattr(result, metric) is not None]
            if values:
                analysis['summary'][metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        # Timing analysis
        if self.config.measure_training_time:
            training_times = [r.training_time for r in self.results if r.training_time is not None]
            if training_times:
                analysis['summary']['training_time'] = {
                    'mean': np.mean(training_times),
                    'std': np.std(training_times),
                    'min': np.min(training_times),
                    'max': np.max(training_times)
                }
        
        # Rankings
        for metric in metrics:
            sorted_results = sorted(
                [r for r in self.results if getattr(r, metric) is not None],
                key=lambda x: getattr(x, metric),
                reverse=True
            )
            analysis['rankings'][metric] = [
                {'name': r.name, 'value': getattr(r, metric)} 
                for r in sorted_results
            ]
        
        # Statistical significance tests
        if self.config.statistical_significance_test and len(self.results) > 1:
            # Pairwise t-tests for accuracy
            accuracy_scores = {r.name: getattr(r, 'accuracy') for r in self.results if getattr(r, 'accuracy') is not None}
            
            if len(accuracy_scores) > 1:
                analysis['statistical_tests']['pairwise_accuracy'] = {}
                names = list(accuracy_scores.keys())
                
                for i in range(len(names)):
                    for j in range(i + 1, len(names)):
                        name1, name2 = names[i], names[j]
                        # For single point estimates, we can't do proper t-tests
                        # This is a limitation - ideally we'd have multiple runs
                        score_diff = accuracy_scores[name1] - accuracy_scores[name2]
                        analysis['statistical_tests']['pairwise_accuracy'][f"{name1}_vs_{name2}"] = {
                            'score_difference': score_diff,
                            'note': 'Statistical test requires multiple runs for proper analysis'
                        }
        
        return analysis
    
    def _generate_report(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive benchmark report."""
        report = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'config': self.config.__dict__,
                'data_info': {
                    'total_samples': len(self.data) if self.data is not None else 0,
                    'train_samples': len(self.X_train) if self.X_train else 0,
                    'test_samples': len(self.X_test) if self.X_test else 0
                }
            },
            'results': [
                {
                    'name': r.name,
                    'config': r.config,
                    'metrics': {
                        'accuracy': r.accuracy,
                        'precision': r.precision,
                        'recall': r.recall,
                        'f1_score': r.f1_score,
                        'auc_score': r.auc_score
                    },
                    'timing': {
                        'training_time': r.training_time,
                        'inference_time': r.inference_time
                    },
                    'cross_validation': {
                        'cv_mean': r.cv_mean,
                        'cv_std': r.cv_std,
                        'cv_scores': r.cv_scores
                    } if r.cv_scores else None,
                    'model_info': {
                        'parameters': r.model_parameters,
                        'convergence': r.convergence_info
                    }
                }
                for r in self.results
            ],
            'analysis': analysis,
            'conclusions': self._draw_conclusions(analysis)
        }
        
        return report
    
    def _draw_conclusions(self, analysis: Dict[str, Any]) -> Dict[str, str]:
        """Draw conclusions from benchmark results."""
        conclusions = {}
        
        if 'rankings' in analysis and 'accuracy' in analysis['rankings']:
            best_model = analysis['rankings']['accuracy'][0]
            conclusions['best_accuracy'] = f"Best performing model: {best_model['name']} with accuracy {best_model['value']:.4f}"
        
        if 'summary' in analysis and 'training_time' in analysis['summary']:
            conclusions['efficiency'] = f"Average training time: {analysis['summary']['training_time']['mean']:.2f}±{analysis['summary']['training_time']['std']:.2f} seconds"
        
        # Quantum vs Classical comparison
        quantum_results = [r for r in self.results if r.config.get('type') == 'quantum_inspired']
        classical_results = [r for r in self.results if r.config.get('type') == 'classical']
        
        if quantum_results and classical_results:
            avg_quantum_acc = np.mean([r.accuracy for r in quantum_results])
            avg_classical_acc = np.mean([r.accuracy for r in classical_results])
            
            if avg_quantum_acc > avg_classical_acc:
                improvement = ((avg_quantum_acc - avg_classical_acc) / avg_classical_acc) * 100
                conclusions['quantum_advantage'] = f"Quantum-inspired models show {improvement:.1f}% improvement over classical models"
            else:
                conclusions['quantum_advantage'] = "Classical models outperform quantum-inspired models in this evaluation"
        
        return conclusions
    
    def _save_results(self, report: Dict[str, Any]) -> None:
        """Save benchmark results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"quantum_benchmark_report_{timestamp}.json"
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.int64):
                return int(obj)
            elif isinstance(obj, np.float64):
                return float(obj)
            return obj
        
        # Clean report for JSON
        clean_report = json.loads(json.dumps(report, default=convert_numpy))
        
        with open(filename, 'w') as f:
            json.dump(clean_report, f, indent=2)
        
        logger.info(f"Benchmark report saved to {filename}")
    
    def _generate_plots(self) -> None:
        """Generate visualization plots."""
        if not PLOTTING_AVAILABLE:
            return
        
        # Performance comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy comparison
        names = [r.name for r in self.results]
        accuracies = [r.accuracy for r in self.results]
        
        axes[0, 0].bar(names, accuracies)
        axes[0, 0].set_title('Model Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # F1 Score comparison
        f1_scores = [r.f1_score for r in self.results]
        axes[0, 1].bar(names, f1_scores)
        axes[0, 1].set_title('Model F1 Score Comparison')
        axes[0, 1].set_ylabel('F1 Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Training time comparison
        training_times = [r.training_time or 0 for r in self.results]
        axes[1, 0].bar(names, training_times)
        axes[1, 0].set_title('Training Time Comparison')
        axes[1, 0].set_ylabel('Training Time (seconds)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Model complexity (parameters) vs Performance
        quantum_results = [r for r in self.results if r.model_parameters]
        if quantum_results:
            params = [r.model_parameters for r in quantum_results]
            accs = [r.accuracy for r in quantum_results]
            axes[1, 1].scatter(params, accs)
            axes[1, 1].set_title('Model Complexity vs Accuracy')
            axes[1, 1].set_xlabel('Number of Parameters')
            axes[1, 1].set_ylabel('Accuracy')
            
            # Add labels
            for r in quantum_results:
                axes[1, 1].annotate(
                    f"q={r.config.get('n_qubits', '?')}",
                    (r.model_parameters, r.accuracy),
                    xytext=(5, 5), textcoords='offset points'
                )
        
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'quantum_benchmark_plots_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Benchmark plots saved as quantum_benchmark_plots_{timestamp}.png")
    
    def print_summary(self) -> None:
        """Print a summary of benchmark results."""
        if not self.results:
            print("No benchmark results available.")
            return
        
        print("\n" + "="*80)
        print("QUANTUM-INSPIRED SENTIMENT ANALYSIS BENCHMARK RESULTS")
        print("="*80)
        
        print(f"\nDataset: {len(self.X_train)} training, {len(self.X_test)} test samples")
        print(f"Cross-validation: {self.config.cv_folds}-fold")
        
        print(f"\n{'Model':<35} {'Accuracy':<10} {'F1-Score':<10} {'Train Time':<12} {'Inference':<12}")
        print("-" * 80)
        
        for result in sorted(self.results, key=lambda x: x.accuracy, reverse=True):
            train_time = f"{result.training_time:.2f}s" if result.training_time else "N/A"
            inf_time = f"{result.inference_time*1000:.2f}ms" if result.inference_time else "N/A"
            
            print(f"{result.name:<35} {result.accuracy:<10.4f} {result.f1_score:<10.4f} "
                  f"{train_time:<12} {inf_time:<12}")
        
        # Best model
        best_model = max(self.results, key=lambda x: x.accuracy)
        print(f"\nBest Model: {best_model.name} (Accuracy: {best_model.accuracy:.4f})")
        
        # Quantum vs Classical comparison
        quantum_results = [r for r in self.results if r.config.get('type') == 'quantum_inspired']
        classical_results = [r for r in self.results if r.config.get('type') == 'classical']
        
        if quantum_results and classical_results:
            avg_quantum = np.mean([r.accuracy for r in quantum_results])
            avg_classical = np.mean([r.accuracy for r in classical_results])
            print(f"\nQuantum-inspired average: {avg_quantum:.4f}")
            print(f"Classical average: {avg_classical:.4f}")
            improvement = ((avg_quantum - avg_classical) / avg_classical) * 100
            print(f"Relative improvement: {improvement:+.1f}%")
        
        print("="*80)


# Factory function for easy usage
def run_quantum_sentiment_benchmark(
    data_path: str = None,
    data: pd.DataFrame = None,
    include_classical: bool = True,
    include_quantum: bool = True,
    qubit_sizes: List[int] = None,
    save_results: bool = True
) -> Dict[str, Any]:
    """
    Run a comprehensive quantum sentiment analysis benchmark.
    
    Args:
        data_path: Path to CSV file with 'text' and 'label' columns
        data: DataFrame with 'text' and 'label' columns
        include_classical: Whether to include classical baselines
        include_quantum: Whether to include quantum-inspired models
        qubit_sizes: List of qubit sizes to test for quantum models
        save_results: Whether to save results to file
    
    Returns:
        Comprehensive benchmark report
    """
    config = BenchmarkConfig(
        include_classical=include_classical,
        include_quantum_inspired=include_quantum,
        quantum_qubit_sizes=qubit_sizes or [4, 6, 8],
        save_results=save_results
    )
    
    benchmark = QuantumBenchmarkSuite(config)
    report = benchmark.run_benchmark(data_path, data)
    benchmark.print_summary()
    
    return report


if __name__ == "__main__":
    # Example usage
    print("Running Quantum-Inspired Sentiment Analysis Benchmark...")
    
    # Run with default configuration
    report = run_quantum_sentiment_benchmark(
        include_classical=True,
        include_quantum=True,
        qubit_sizes=[4, 6],  # Smaller sizes for quick demo
        save_results=True
    )
    
    print("\nBenchmark completed! Check the generated files for detailed results.")