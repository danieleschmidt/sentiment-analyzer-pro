"""
Tests for quantum benchmarking framework.

This module tests the comprehensive benchmarking system for quantum-inspired
sentiment analysis models.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock

from src.quantum_benchmarking import (
    BenchmarkConfig,
    ModelResult,
    QuantumBenchmarkSuite,
    run_quantum_sentiment_benchmark
)


class TestBenchmarkConfig:
    """Test benchmark configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = BenchmarkConfig()
        
        assert config.test_size == 0.2
        assert config.random_state == 42
        assert config.cv_folds == 5
        assert config.include_classical is True
        assert config.include_quantum_inspired is True
        assert config.quantum_qubit_sizes == [4, 6, 8]
        assert config.quantum_layer_depths == [2, 3, 4]
        assert config.alpha == 0.05
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = BenchmarkConfig(
            test_size=0.3,
            cv_folds=3,
            include_classical=False,
            quantum_qubit_sizes=[2, 4],
            alpha=0.01
        )
        
        assert config.test_size == 0.3
        assert config.cv_folds == 3
        assert config.include_classical is False
        assert config.quantum_qubit_sizes == [2, 4]
        assert config.alpha == 0.01


class TestModelResult:
    """Test model result data structure."""
    
    def test_model_result_creation(self):
        """Test creating a model result."""
        result = ModelResult(
            name="Test Model",
            config={"type": "test"},
            accuracy=0.85,
            precision=0.80,
            recall=0.90,
            f1_score=0.85
        )
        
        assert result.name == "Test Model"
        assert result.config == {"type": "test"}
        assert result.accuracy == 0.85
        assert result.precision == 0.80
        assert result.recall == 0.90
        assert result.f1_score == 0.85
        assert result.auc_score is None  # Default
        assert result.training_time is None  # Default
    
    def test_model_result_with_optional_fields(self):
        """Test model result with optional fields."""
        cv_scores = [0.8, 0.85, 0.9, 0.82, 0.88]
        confusion_mat = np.array([[10, 2], [3, 15]])
        
        result = ModelResult(
            name="Advanced Model",
            config={"type": "quantum", "qubits": 4},
            accuracy=0.90,
            precision=0.85,
            recall=0.95,
            f1_score=0.90,
            auc_score=0.92,
            training_time=120.5,
            inference_time=0.002,
            cv_scores=cv_scores,
            cv_mean=np.mean(cv_scores),
            cv_std=np.std(cv_scores),
            confusion_matrix=confusion_mat,
            model_parameters=48
        )
        
        assert result.auc_score == 0.92
        assert result.training_time == 120.5
        assert result.inference_time == 0.002
        assert result.cv_scores == cv_scores
        assert result.cv_mean == np.mean(cv_scores)
        assert result.cv_std == np.std(cv_scores)
        assert np.array_equal(result.confusion_matrix, confusion_mat)
        assert result.model_parameters == 48


class TestQuantumBenchmarkSuite:
    """Test the main benchmarking suite."""
    
    def test_initialization(self):
        """Test benchmark suite initialization."""
        config = BenchmarkConfig(cv_folds=3)
        suite = QuantumBenchmarkSuite(config)
        
        assert suite.config == config
        assert suite.results == []
        assert suite.data is None
        assert suite.X_train is None
    
    def test_initialization_with_default_config(self):
        """Test initialization with default config."""
        suite = QuantumBenchmarkSuite()
        
        assert isinstance(suite.config, BenchmarkConfig)
        assert suite.config.cv_folds == 5  # Default value
    
    def test_create_synthetic_data(self):
        """Test synthetic data creation."""
        suite = QuantumBenchmarkSuite()
        data = suite._create_synthetic_data()
        
        assert isinstance(data, pd.DataFrame)
        assert 'text' in data.columns
        assert 'label' in data.columns
        assert len(data) == 100  # 50 positive + 50 negative
        
        # Check label distribution
        label_counts = data['label'].value_counts()
        assert label_counts['positive'] == 50
        assert label_counts['negative'] == 50
        
        # Check that texts are strings
        assert all(isinstance(text, str) for text in data['text'])
        assert all(label in ['positive', 'negative'] for label in data['label'])
    
    def test_load_data_from_dataframe(self):
        """Test loading data from DataFrame."""
        # Create test data
        test_data = pd.DataFrame({
            'text': ['I love this', 'This is bad', 'Great product', 'Terrible service'],
            'label': ['positive', 'negative', 'positive', 'negative']
        })
        
        suite = QuantumBenchmarkSuite()
        suite.load_data(data=test_data)
        
        assert suite.data is not None
        assert len(suite.data) == 4
        assert 'text_clean' in suite.data.columns
        assert suite.X_train is not None
        assert suite.X_test is not None
        assert suite.y_train is not None
        assert suite.y_test is not None
        
        # Check train/test split
        total_samples = len(suite.X_train) + len(suite.X_test)
        assert total_samples == 4
    
    def test_load_data_synthetic(self):
        """Test loading synthetic data."""
        suite = QuantumBenchmarkSuite()
        suite.load_data()  # No data provided, should create synthetic
        
        assert suite.data is not None
        assert len(suite.data) == 100
        assert suite.X_train is not None
        assert len(suite.X_train) == 80  # 80% of 100
        assert len(suite.X_test) == 20  # 20% of 100
    
    def test_load_data_missing_columns_raises_error(self):
        """Test that missing columns raise error."""
        bad_data = pd.DataFrame({
            'content': ['Some text'],  # Wrong column name
            'sentiment': ['positive']  # Wrong column name
        })
        
        suite = QuantumBenchmarkSuite()
        
        with pytest.raises(ValueError, match="Data must contain 'text' and 'label' columns"):
            suite.load_data(data=bad_data)
    
    @patch('src.quantum_benchmarking.build_model')
    @patch('src.quantum_benchmarking.build_nb_model')
    def test_benchmark_classical_models(self, mock_nb_model, mock_lr_model):
        """Test benchmarking classical models."""
        # Mock the models
        mock_lr = Mock()
        mock_lr.fit.return_value = None
        mock_lr.predict.return_value = ['positive', 'negative']
        mock_lr.pipeline = Mock()
        mock_lr_model.return_value = mock_lr
        
        mock_nb = Mock()
        mock_nb.fit.return_value = None
        mock_nb.predict.return_value = ['positive', 'negative']
        mock_nb.pipeline = Mock()
        mock_nb_model.return_value = mock_nb
        
        # Mock cross_val_score
        with patch('src.quantum_benchmarking.cross_val_score') as mock_cv:
            mock_cv.return_value = np.array([0.8, 0.85, 0.9, 0.82, 0.88])
            
            config = BenchmarkConfig(include_classical=True)
            suite = QuantumBenchmarkSuite(config)
            
            # Set up test data
            suite.X_train = ['good', 'bad'] * 10
            suite.X_test = ['positive text', 'negative text']
            suite.y_train = ['positive', 'negative'] * 10
            suite.y_test = ['positive', 'negative']
            
            suite._benchmark_classical_models()
            
            # Check that models were created and fitted
            mock_lr_model.assert_called_once()
            mock_nb_model.assert_called_once()
            mock_lr.fit.assert_called_once()
            mock_nb.fit.assert_called_once()
            
            # Check that results were recorded
            assert len(suite.results) == 2
            
            # Find results by name
            lr_result = next(r for r in suite.results if r.name == "Logistic Regression")
            nb_result = next(r for r in suite.results if r.name == "Naive Bayes")
            
            assert lr_result.config['type'] == 'classical'
            assert nb_result.config['type'] == 'classical'
            assert lr_result.accuracy is not None
            assert nb_result.accuracy is not None
    
    @patch('src.quantum_benchmarking.QuantumInspiredSentimentClassifier')
    def test_benchmark_quantum_inspired_models(self, mock_classifier_class):
        """Test benchmarking quantum-inspired models."""
        # Mock the quantum classifier
        mock_classifier = Mock()
        mock_classifier.fit.return_value = None
        mock_classifier.predict.return_value = ['positive', 'negative']
        mock_classifier.predict_proba.return_value = np.array([[0.2, 0.8], [0.9, 0.1]])
        mock_classifier.training_history = [{'final_loss': 0.1, 'iterations': 10, 'success': True}]
        mock_classifier_class.return_value = mock_classifier
        
        config = BenchmarkConfig(
            include_quantum_inspired=True,
            quantum_qubit_sizes=[4],
            quantum_layer_depths=[2]
        )
        suite = QuantumBenchmarkSuite(config)
        
        # Set up test data
        suite.X_train = ['good', 'bad'] * 10
        suite.X_test = ['positive text', 'negative text']
        suite.y_train = ['positive', 'negative'] * 10
        suite.y_test = ['positive', 'negative']
        
        suite._benchmark_quantum_inspired_models()
        
        # Check that quantum model was created and trained
        mock_classifier_class.assert_called()
        mock_classifier.fit.assert_called_once()
        
        # Check that results were recorded
        quantum_results = [r for r in suite.results if r.config.get('type') == 'quantum_inspired']
        assert len(quantum_results) == 1
        
        result = quantum_results[0]
        assert result.name == "Quantum-Inspired (q=4, l=2)"
        assert result.config['n_qubits'] == 4
        assert result.config['n_layers'] == 2
        assert result.accuracy is not None
        assert result.auc_score is not None
    
    def test_analyze_results_empty(self):
        """Test analyzing results when no results exist."""
        suite = QuantumBenchmarkSuite()
        analysis = suite._analyze_results()
        
        assert analysis == {}
    
    def test_analyze_results_with_data(self):
        """Test analyzing results with sample data."""
        suite = QuantumBenchmarkSuite()
        
        # Add sample results
        suite.results = [
            ModelResult(
                name="Model 1",
                config={"type": "classical"},
                accuracy=0.85,
                precision=0.80,
                recall=0.90,
                f1_score=0.85,
                training_time=10.5
            ),
            ModelResult(
                name="Model 2",
                config={"type": "quantum_inspired"},
                accuracy=0.90,
                precision=0.88,
                recall=0.92,
                f1_score=0.90,
                training_time=25.0
            )
        ]
        
        analysis = suite._analyze_results()
        
        assert 'summary' in analysis
        assert 'rankings' in analysis
        
        # Check summary statistics
        assert 'accuracy' in analysis['summary']
        assert analysis['summary']['accuracy']['mean'] == 0.875  # (0.85 + 0.90) / 2
        
        # Check rankings
        assert 'accuracy' in analysis['rankings']
        best_accuracy = analysis['rankings']['accuracy'][0]
        assert best_accuracy['name'] == "Model 2"
        assert best_accuracy['value'] == 0.90
    
    def test_generate_report(self):
        """Test report generation."""
        suite = QuantumBenchmarkSuite()
        
        # Add sample data and results
        suite.data = pd.DataFrame({'text': ['test'], 'label': ['positive']})
        suite.X_train = ['train']
        suite.X_test = ['test']
        
        suite.results = [
            ModelResult(
                name="Test Model",
                config={"type": "test"},
                accuracy=0.85,
                precision=0.80,
                recall=0.90,
                f1_score=0.85
            )
        ]
        
        analysis = {'summary': {}, 'rankings': {}}
        report = suite._generate_report(analysis)
        
        assert 'metadata' in report
        assert 'results' in report
        assert 'analysis' in report
        assert 'conclusions' in report
        
        # Check metadata
        assert 'timestamp' in report['metadata']
        assert 'config' in report['metadata']
        assert 'data_info' in report['metadata']
        
        # Check results
        assert len(report['results']) == 1
        result = report['results'][0]
        assert result['name'] == "Test Model"
        assert result['metrics']['accuracy'] == 0.85
    
    def test_draw_conclusions(self):
        """Test drawing conclusions from analysis."""
        suite = QuantumBenchmarkSuite()
        
        analysis = {
            'rankings': {
                'accuracy': [
                    {'name': 'Best Model', 'value': 0.95},
                    {'name': 'Second Model', 'value': 0.90}
                ]
            },
            'summary': {
                'training_time': {
                    'mean': 15.5,
                    'std': 5.2
                }
            }
        }
        
        # Add some results for comparison
        suite.results = [
            ModelResult(
                name="Classical Model",
                config={"type": "classical"},
                accuracy=0.85,
                precision=0.80,
                recall=0.90,
                f1_score=0.85
            ),
            ModelResult(
                name="Quantum Model",
                config={"type": "quantum_inspired"},
                accuracy=0.90,
                precision=0.88,
                recall=0.92,
                f1_score=0.90
            )
        ]
        
        conclusions = suite._draw_conclusions(analysis)
        
        assert 'best_accuracy' in conclusions
        assert 'efficiency' in conclusions
        assert 'quantum_advantage' in conclusions
        
        assert "Best Model" in conclusions['best_accuracy']
        assert "15.5" in conclusions['efficiency']
        assert "improvement" in conclusions['quantum_advantage'].lower()
    
    @patch('src.quantum_benchmarking.json.dump')
    def test_save_results(self, mock_json_dump):
        """Test saving results to file."""
        suite = QuantumBenchmarkSuite()
        
        report = {
            'metadata': {'timestamp': '2024-01-01'},
            'results': [],
            'analysis': {},
            'conclusions': {}
        }
        
        suite._save_results(report)
        
        # Check that json.dump was called
        mock_json_dump.assert_called_once()
        call_args = mock_json_dump.call_args
        assert call_args[0][0] == report  # First argument should be the report
    
    def test_print_summary_no_results(self, capsys):
        """Test printing summary with no results."""
        suite = QuantumBenchmarkSuite()
        suite.print_summary()
        
        captured = capsys.readouterr()
        assert "No benchmark results available" in captured.out
    
    def test_print_summary_with_results(self, capsys):
        """Test printing summary with results."""
        suite = QuantumBenchmarkSuite()
        
        # Set up test data
        suite.X_train = ['train'] * 80
        suite.X_test = ['test'] * 20
        
        suite.results = [
            ModelResult(
                name="Test Model",
                config={"type": "classical"},
                accuracy=0.85,
                precision=0.80,
                recall=0.90,
                f1_score=0.85,
                training_time=10.5,
                inference_time=0.002
            )
        ]
        
        suite.print_summary()
        
        captured = capsys.readouterr()
        assert "QUANTUM-INSPIRED SENTIMENT ANALYSIS BENCHMARK RESULTS" in captured.out
        assert "Test Model" in captured.out
        assert "0.8500" in captured.out  # Accuracy
        assert "80 training, 20 test samples" in captured.out


class TestFactoryFunction:
    """Test the factory function for running benchmarks."""
    
    @patch('src.quantum_benchmarking.QuantumBenchmarkSuite')
    def test_run_quantum_sentiment_benchmark(self, mock_suite_class):
        """Test the factory function."""
        # Mock the benchmark suite
        mock_suite = Mock()
        mock_report = {'test': 'report'}
        mock_suite.run_benchmark.return_value = mock_report
        mock_suite.print_summary.return_value = None
        mock_suite_class.return_value = mock_suite
        
        # Call factory function
        result = run_quantum_sentiment_benchmark(
            include_classical=True,
            include_quantum=True,
            qubit_sizes=[4, 6],
            save_results=False
        )
        
        # Check that suite was created and used
        mock_suite_class.assert_called_once()
        mock_suite.run_benchmark.assert_called_once()
        mock_suite.print_summary.assert_called_once()
        
        assert result == mock_report
    
    def test_run_quantum_sentiment_benchmark_config(self):
        """Test that factory function creates correct config."""
        with patch('src.quantum_benchmarking.QuantumBenchmarkSuite') as mock_suite_class:
            mock_suite = Mock()
            mock_suite.run_benchmark.return_value = {}
            mock_suite_class.return_value = mock_suite
            
            run_quantum_sentiment_benchmark(
                include_classical=False,
                include_quantum=True,
                qubit_sizes=[2, 4],
                save_results=True
            )
            
            # Check that config was created with correct parameters
            call_args = mock_suite_class.call_args[0][0]  # First argument is config
            assert call_args.include_classical is False
            assert call_args.include_quantum_inspired is True
            assert call_args.quantum_qubit_sizes == [2, 4]
            assert call_args.save_results is True


class TestIntegration:
    """Integration tests for the benchmarking system."""
    
    def test_minimal_benchmark_run(self):
        """Test a minimal benchmark run with real components."""
        # Create a minimal dataset
        data = pd.DataFrame({
            'text': ['I love this', 'This is bad', 'Great product', 'Poor service'],
            'label': ['positive', 'negative', 'positive', 'negative']
        })
        
        # Configure for quick run
        config = BenchmarkConfig(
            include_classical=True,
            include_quantum_inspired=True,
            quantum_qubit_sizes=[3],  # Small for speed
            quantum_layer_depths=[1],  # Single layer for speed
            save_results=False,
            generate_plots=False
        )
        
        suite = QuantumBenchmarkSuite(config)
        
        # This would be a real integration test, but might be slow
        # For now, just test that the suite can be set up correctly
        suite.load_data(data=data)
        
        assert suite.data is not None
        assert len(suite.X_train) > 0
        assert len(suite.X_test) > 0
        
        # Test that analysis works with empty results
        analysis = suite._analyze_results()
        assert isinstance(analysis, dict)
    
    def test_benchmark_configuration_validation(self):
        """Test various benchmark configurations."""
        # Test different configurations don't break initialization
        configs = [
            BenchmarkConfig(include_classical=False, include_quantum_inspired=True),
            BenchmarkConfig(include_classical=True, include_quantum_inspired=False),
            BenchmarkConfig(quantum_qubit_sizes=[2], quantum_layer_depths=[1]),
            BenchmarkConfig(cv_folds=3, test_size=0.3),
        ]
        
        for config in configs:
            suite = QuantumBenchmarkSuite(config)
            assert suite.config == config
            
            # Test that synthetic data generation works
            suite.load_data()
            assert suite.data is not None


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])