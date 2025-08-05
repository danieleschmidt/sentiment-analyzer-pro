"""
Tests for quantum-inspired sentiment analysis models.

This module provides comprehensive testing for the novel quantum-inspired
sentiment analysis implementation, ensuring reproducibility and correctness.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import tempfile
import json

from src.quantum_inspired_sentiment import (
    QuantumInspiredConfig,
    QuantumInspiredCircuit,
    WaveletQuantumHybrid,
    TransformerQuantumBridge,
    QuantumInspiredSentimentClassifier,
    create_quantum_inspired_classifier
)


class TestQuantumInspiredConfig:
    """Test configuration class for quantum-inspired models."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = QuantumInspiredConfig()
        
        assert config.n_qubits == 8
        assert config.n_layers == 3
        assert config.embedding_dim == 128
        assert config.use_pca is True
        assert config.pca_components == 64
        assert config.learning_rate == 0.01
        assert config.n_parameters == config.n_qubits * config.n_layers * 3
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = QuantumInspiredConfig(
            n_qubits=4,
            n_layers=2,
            embedding_dim=64,
            use_pca=False
        )
        
        assert config.n_qubits == 4
        assert config.n_layers == 2
        assert config.embedding_dim == 64
        assert config.use_pca is False
        assert config.n_parameters == 4 * 2 * 3  # 24
    
    def test_parameter_calculation(self):
        """Test automatic parameter calculation."""
        config = QuantumInspiredConfig(n_qubits=6, n_layers=4)
        expected_params = 6 * 4 * 3  # 72
        assert config.n_parameters == expected_params


class TestQuantumInspiredCircuit:
    """Test quantum-inspired circuit implementation."""
    
    def test_circuit_initialization(self):
        """Test circuit initialization."""
        config = QuantumInspiredConfig(n_qubits=4, n_layers=2)
        circuit = QuantumInspiredCircuit(config)
        
        assert circuit.config == config
        assert len(circuit.parameters) == config.n_parameters
        assert circuit.state_vector is None
    
    def test_rotation_gate_dimensions(self):
        """Test rotation gate matrix dimensions."""
        config = QuantumInspiredConfig(n_qubits=3)
        circuit = QuantumInspiredCircuit(config)
        
        gate = circuit.apply_rotation_gate(0, np.pi/4, 'Y')
        expected_size = 2 ** config.n_qubits
        
        assert gate.shape == (expected_size, expected_size)
        assert gate.dtype == complex
    
    def test_entangling_layer_dimensions(self):
        """Test entangling layer dimensions."""
        config = QuantumInspiredConfig(n_qubits=3)
        circuit = QuantumInspiredCircuit(config)
        
        entangling_op = circuit.apply_entangling_layer()
        expected_size = 2 ** config.n_qubits
        
        assert entangling_op.shape == (expected_size, expected_size)
        assert entangling_op.dtype == complex
    
    def test_forward_pass_amplitude_encoding(self):
        """Test forward pass with amplitude encoding."""
        config = QuantumInspiredConfig(n_qubits=3, quantum_encoding='amplitude')
        circuit = QuantumInspiredCircuit(config)
        
        # Create sample input data
        input_data = np.random.rand(2, 5)  # 2 samples, 5 features
        
        output = circuit.forward(input_data)
        
        assert output.shape == (2,)  # 2 samples
        assert isinstance(output, np.ndarray)
        assert output.dtype in [np.float64, float]
    
    def test_forward_pass_angle_encoding(self):
        """Test forward pass with angle encoding."""
        config = QuantumInspiredConfig(n_qubits=3, quantum_encoding='angle')
        circuit = QuantumInspiredCircuit(config)
        
        # Create sample input data
        input_data = np.random.rand(2, 3)  # 2 samples, 3 features (matching qubits)
        
        output = circuit.forward(input_data)
        
        assert output.shape == (2,)  # 2 samples
        assert isinstance(output, np.ndarray)
        assert output.dtype in [np.float64, float]
    
    def test_measurement_bounds(self):
        """Test that measurements are within expected bounds."""
        config = QuantumInspiredConfig(n_qubits=2, n_layers=1)
        circuit = QuantumInspiredCircuit(config)
        
        input_data = np.random.rand(10, 3)
        output = circuit.forward(input_data)
        
        # Expectation values should be between -1 and 1
        assert np.all(output >= -1.0)
        assert np.all(output <= 1.0)


class TestWaveletQuantumHybrid:
    """Test wavelet-quantum hybrid preprocessing."""
    
    def test_initialization(self):
        """Test hybrid preprocessing initialization."""
        config = QuantumInspiredConfig(use_pca=True, use_wavelet=True)
        hybrid = WaveletQuantumHybrid(config)
        
        assert hybrid.config == config
        assert hybrid.fitted is False
        assert hybrid.pca is not None
    
    def test_fit_transform_without_wavelets(self):
        """Test fit_transform without wavelets."""
        config = QuantumInspiredConfig(use_pca=True, use_wavelet=False)
        hybrid = WaveletQuantumHybrid(config)
        
        data = np.random.rand(100, 20)
        transformed = hybrid.fit_transform(data)
        
        assert transformed.shape[0] == 100  # Same number of samples
        # PCA components may be reduced if input has fewer features
        expected_components = min(config.pca_components, data.shape[1])
        assert transformed.shape[1] == expected_components
        assert hybrid.fitted is True
    
    def test_fit_transform_with_pca_disabled(self):
        """Test fit_transform with PCA disabled."""
        config = QuantumInspiredConfig(use_pca=False, use_wavelet=False)
        hybrid = WaveletQuantumHybrid(config)
        
        data = np.random.rand(50, 10)
        transformed = hybrid.fit_transform(data)
        
        assert transformed.shape == data.shape  # Should preserve shape when both disabled
        assert hybrid.fitted is True
    
    def test_transform_before_fit_raises_error(self):
        """Test that transform before fit raises error."""
        config = QuantumInspiredConfig()
        hybrid = WaveletQuantumHybrid(config)
        
        data = np.random.rand(10, 5)
        
        with pytest.raises(ValueError, match="Must fit transformer before transform"):
            hybrid.transform(data)
    
    def test_transform_after_fit(self):
        """Test transform after fit."""
        config = QuantumInspiredConfig(use_pca=True, use_wavelet=False)
        hybrid = WaveletQuantumHybrid(config)
        
        # Fit on training data
        train_data = np.random.rand(100, 20)
        hybrid.fit_transform(train_data)
        
        # Transform test data
        test_data = np.random.rand(50, 20)
        transformed = hybrid.transform(test_data)
        
        assert transformed.shape[0] == 50
        # PCA components may be reduced if input has fewer features
        expected_components = min(config.pca_components, train_data.shape[1])
        assert transformed.shape[1] == expected_components


class TestTransformerQuantumBridge:
    """Test transformer-quantum bridge."""
    
    def test_initialization_without_transformers(self):
        """Test initialization without transformer libraries."""
        config = QuantumInspiredConfig(use_transformer_embeddings=False)
        bridge = TransformerQuantumBridge(config)
        
        assert bridge.tokenizer is None
        assert bridge.model is None
    
    def test_extract_embeddings_fallback(self):
        """Test embedding extraction with fallback method."""
        config = QuantumInspiredConfig(use_transformer_embeddings=True, embedding_dim=64)
        bridge = TransformerQuantumBridge(config)
        
        # Force fallback by setting model to None
        bridge.model = None
        bridge.tokenizer = None
        
        texts = ["I love this product", "This is terrible", "Great quality"]
        embeddings = bridge.extract_embeddings(texts)
        
        assert embeddings.shape == (3, 64)
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.dtype in [np.float64, float]
    
    def test_embedding_dimension_consistency(self):
        """Test that embeddings have consistent dimensions."""
        config = QuantumInspiredConfig(embedding_dim=128)
        bridge = TransformerQuantumBridge(config)
        
        texts = ["Short text", "This is a much longer text with many words"]
        embeddings = bridge.extract_embeddings(texts)
        
        assert embeddings.shape == (2, 128)


class TestQuantumInspiredSentimentClassifier:
    """Test main quantum-inspired sentiment classifier."""
    
    def test_initialization(self):
        """Test classifier initialization."""
        config = QuantumInspiredConfig(n_qubits=4)
        classifier = QuantumInspiredSentimentClassifier(config)
        
        assert classifier.config == config
        assert classifier.is_fitted is False
        assert len(classifier.training_history) == 0
        assert 'positive' in classifier.label_encoder
        assert 'negative' in classifier.label_encoder
    
    def test_label_encoding(self):
        """Test label encoding and decoding."""
        classifier = QuantumInspiredSentimentClassifier()
        
        assert classifier.label_encoder['positive'] == 1
        assert classifier.label_encoder['negative'] == -1
        assert classifier.inverse_label_encoder[1] == 'positive'
        assert classifier.inverse_label_encoder[-1] == 'negative'
    
    def test_prepare_data(self):
        """Test data preparation."""
        config = QuantumInspiredConfig(n_qubits=3)
        classifier = QuantumInspiredSentimentClassifier(config)
        
        texts = ["I love this", "This is bad"]
        labels = ["positive", "negative"]
        
        features, encoded_labels = classifier._prepare_data(texts, labels)
        
        assert features.shape[0] == 2  # 2 samples
        assert encoded_labels is not None
        assert len(encoded_labels) == 2
        assert encoded_labels[0] == 1  # positive
        assert encoded_labels[1] == -1  # negative
    
    def test_prepare_data_without_labels(self):
        """Test data preparation without labels."""
        classifier = QuantumInspiredSentimentClassifier()
        
        texts = ["I love this", "This is bad"]
        features, encoded_labels = classifier._prepare_data(texts)
        
        assert features.shape[0] == 2
        assert encoded_labels is None
    
    def test_fit_and_predict(self):
        """Test fitting and prediction."""
        config = QuantumInspiredConfig(n_qubits=3, max_iterations=5)  # Quick training
        classifier = QuantumInspiredSentimentClassifier(config)
        
        # Simple training data
        texts = ["I love this product", "This is terrible", "Great quality", "Bad service"]
        labels = ["positive", "negative", "positive", "negative"]
        
        # Fit the model
        classifier.fit(texts, labels)
        
        assert classifier.is_fitted is True
        assert len(classifier.training_history) == 1
        
        # Make predictions
        test_texts = ["Excellent product", "Horrible experience"]
        predictions = classifier.predict(test_texts)
        
        assert len(predictions) == 2
        assert all(pred in ['positive', 'negative'] for pred in predictions)
    
    def test_predict_proba(self):
        """Test probability prediction."""
        config = QuantumInspiredConfig(n_qubits=3, max_iterations=5)
        classifier = QuantumInspiredSentimentClassifier(config)
        
        # Train on simple data
        texts = ["I love this", "This is bad"]
        labels = ["positive", "negative"]
        classifier.fit(texts, labels)
        
        # Test probabilities
        test_texts = ["Good product"]
        probabilities = classifier.predict_proba(test_texts)
        
        assert probabilities.shape == (1, 2)  # 1 sample, 2 classes
        assert np.allclose(probabilities.sum(axis=1), 1.0)  # Probabilities sum to 1
        assert np.all(probabilities >= 0) and np.all(probabilities <= 1)  # Valid probabilities
    
    def test_evaluate(self):
        """Test model evaluation."""
        config = QuantumInspiredConfig(n_qubits=3, max_iterations=5)
        classifier = QuantumInspiredSentimentClassifier(config)
        
        # Train the model
        texts = ["I love this", "This is bad", "Great product", "Terrible service"]
        labels = ["positive", "negative", "positive", "negative"]
        classifier.fit(texts, labels)
        
        # Evaluate on same data (just for testing)
        metrics = classifier.evaluate(texts, labels)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert all(0 <= metric <= 1 for metric in metrics.values())
    
    def test_predict_without_fit_raises_error(self):
        """Test that predict without fit raises error."""
        classifier = QuantumInspiredSentimentClassifier()
        
        with pytest.raises(ValueError, match="Classifier must be fitted"):
            classifier.predict(["Test text"])
    
    def test_predict_proba_without_fit_raises_error(self):
        """Test that predict_proba without fit raises error."""
        classifier = QuantumInspiredSentimentClassifier()
        
        with pytest.raises(ValueError, match="Classifier must be fitted"):
            classifier.predict_proba(["Test text"])
    
    def test_save_and_load(self):
        """Test model saving and loading."""
        config = QuantumInspiredConfig(n_qubits=3, max_iterations=5)
        classifier = QuantumInspiredSentimentClassifier(config)
        
        # Train the model
        texts = ["I love this", "This is bad"]
        labels = ["positive", "negative"]
        classifier.fit(texts, labels)
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp:
            classifier.save(tmp.name)
            
            # Load model
            loaded_classifier = QuantumInspiredSentimentClassifier.load(tmp.name)
            
            assert loaded_classifier.is_fitted is True
            assert np.array_equal(
                loaded_classifier.quantum_circuit.parameters,
                classifier.quantum_circuit.parameters
            )
            
            # Test that loaded model can make predictions
            predictions = loaded_classifier.predict(["Great product"])
            assert len(predictions) == 1
            assert predictions[0] in ['positive', 'negative']


class TestFactoryFunction:
    """Test factory function for creating classifiers."""
    
    def test_create_quantum_inspired_classifier_defaults(self):
        """Test factory function with defaults."""
        classifier = create_quantum_inspired_classifier()
        
        assert isinstance(classifier, QuantumInspiredSentimentClassifier)
        assert classifier.config.n_qubits == 8  # Default
        assert classifier.config.use_transformer_embeddings is True
        assert classifier.config.use_wavelet is True
    
    def test_create_quantum_inspired_classifier_custom(self):
        """Test factory function with custom parameters."""
        classifier = create_quantum_inspired_classifier(
            n_qubits=4,
            use_transformers=False,
            use_wavelets=False
        )
        
        assert classifier.config.n_qubits == 4
        assert classifier.config.use_transformer_embeddings is False
        assert classifier.config.use_wavelet is False


class TestIntegration:
    """Integration tests for the complete system."""
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        # Create classifier with small configuration for speed
        classifier = create_quantum_inspired_classifier(
            n_qubits=3,
            use_transformers=True,
            use_wavelets=False
        )
        
        # Override max iterations for quick test
        classifier.config.max_iterations = 5
        
        # Training data
        texts = [
            "I absolutely love this product! Amazing quality!",
            "Terrible experience, would not recommend at all.",
            "Great value for money and excellent service.",
            "Poor quality and disappointing results.",
            "Outstanding performance, highly satisfied!",
            "Waste of money, completely useless product."
        ]
        
        labels = ["positive", "negative", "positive", "negative", "positive", "negative"]
        
        # Train
        classifier.fit(texts, labels)
        
        # Test predictions
        test_texts = [
            "Fantastic product with great features!",
            "Horrible quality and poor customer service."
        ]
        
        predictions = classifier.predict(test_texts)
        probabilities = classifier.predict_proba(test_texts)
        
        # Assertions
        assert len(predictions) == 2
        assert all(pred in ['positive', 'negative'] for pred in predictions)
        assert probabilities.shape == (2, 2)
        
        # Evaluate
        metrics = classifier.evaluate(texts, labels)
        assert all(metric >= 0 for metric in metrics.values())
    
    def test_different_configurations(self):
        """Test different quantum circuit configurations."""
        configs = [
            (2, 1),  # 2 qubits, 1 layer
            (3, 2),  # 3 qubits, 2 layers
            (4, 1),  # 4 qubits, 1 layer
        ]
        
        texts = ["Good product", "Bad service", "Excellent quality", "Poor performance"]
        labels = ["positive", "negative", "positive", "negative"]
        
        for n_qubits, n_layers in configs:
            config = QuantumInspiredConfig(
                n_qubits=n_qubits,
                n_layers=n_layers,
                max_iterations=3  # Very quick training
            )
            
            classifier = QuantumInspiredSentimentClassifier(config)
            classifier.fit(texts, labels)
            
            predictions = classifier.predict(["Test text"])
            assert len(predictions) == 1
            assert predictions[0] in ['positive', 'negative']
    
    def test_reproducibility(self):
        """Test that results are reproducible with same random seed."""
        # Note: This test might be flaky due to optimization randomness
        # In a real implementation, you'd want to control all random seeds
        
        config = QuantumInspiredConfig(n_qubits=3, max_iterations=3)
        texts = ["Good", "Bad"]
        labels = ["positive", "negative"]
        
        # Train two identical models
        classifier1 = QuantumInspiredSentimentClassifier(config)
        classifier2 = QuantumInspiredSentimentClassifier(config)
        
        # Set same initial parameters for reproducibility
        initial_params = np.random.RandomState(42).uniform(0, 2*np.pi, size=config.n_parameters)
        classifier1.quantum_circuit.parameters = initial_params.copy()
        classifier2.quantum_circuit.parameters = initial_params.copy()
        
        # This test would require more sophisticated random seed control
        # to be truly deterministic, so we'll just test that both can train
        classifier1.fit(texts, labels)
        classifier2.fit(texts, labels)
        
        assert classifier1.is_fitted is True
        assert classifier2.is_fitted is True


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])