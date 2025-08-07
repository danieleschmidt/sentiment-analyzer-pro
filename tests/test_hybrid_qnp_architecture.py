"""
Comprehensive Test Suite for Hybrid Quantum-Neuromorphic-Photonic Architecture
==============================================================================

Tests the novel QNP architecture implementation including:
- Individual modality processors
- Cross-modal bridges and interfaces  
- Triadic fusion mechanisms
- Research analysis capabilities
- Performance benchmarking
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock
import tempfile
import json
from typing import Dict, Any, List

# Import QNP modules
from src.hybrid_qnp_architecture import (
    QNPConfig, FusionMode, HybridQNPArchitecture, QNPSentimentAnalyzer,
    QuantumNeuromorphicBridge, PhotonicQuantumInterface, TriadicFusionLayer,
    create_qnp_analyzer, demonstrate_qnp_breakthrough
)


class TestQNPConfig:
    """Test QNP configuration handling."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = QNPConfig()
        
        assert config.n_qubits == 8
        assert config.quantum_layers == 3
        assert config.spike_timesteps == 100
        assert config.fusion_mode == FusionMode.HIERARCHICAL
        assert config.input_dim == 768
        assert config.output_classes == 3
    
    def test_custom_config(self):
        """Test custom configuration creation."""
        config = QNPConfig(
            n_qubits=12,
            fusion_mode=FusionMode.ADAPTIVE,
            photonic_channels=128,
            enable_coherence_analysis=False
        )
        
        assert config.n_qubits == 12
        assert config.fusion_mode == FusionMode.ADAPTIVE
        assert config.photonic_channels == 128
        assert config.enable_coherence_analysis == False
    
    def test_fusion_modes(self):
        """Test all fusion mode options."""
        modes = [FusionMode.EARLY_FUSION, FusionMode.LATE_FUSION, 
                FusionMode.HIERARCHICAL, FusionMode.ADAPTIVE]
        
        for mode in modes:
            config = QNPConfig(fusion_mode=mode)
            assert config.fusion_mode == mode


class TestQuantumNeuromorphicBridge:
    """Test quantum-neuromorphic bridge functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = QNPConfig(n_qubits=4, neuromorphic_layers=2, hidden_dim=64)
        self.bridge = QuantumNeuromorphicBridge(self.config)
    
    def test_bridge_initialization(self):
        """Test bridge component initialization."""
        assert self.bridge.config == self.config
        assert hasattr(self.bridge, 'state_encoder')
        assert hasattr(self.bridge, 'spike_decoder')
        assert hasattr(self.bridge, 'correlation_matrix')
    
    def test_quantum_to_spikes(self):
        """Test quantum state to spike conversion."""
        batch_size = 3
        n_states = 2 ** self.config.n_qubits
        quantum_states = torch.randn(batch_size, n_states)
        
        spike_probs = self.bridge.quantum_to_spikes(quantum_states, timestep=5)
        
        assert spike_probs.shape == (batch_size, self.config.neuromorphic_layers, self.config.hidden_dim)
        assert torch.all(spike_probs >= 0) and torch.all(spike_probs <= 1)
    
    def test_spikes_to_quantum(self):
        """Test spike train to quantum conversion."""
        batch_size = 3
        spike_trains = torch.rand(batch_size, self.config.neuromorphic_layers, self.config.hidden_dim)
        
        quantum_amplitudes = self.bridge.spikes_to_quantum(spike_trains)
        
        assert quantum_amplitudes.shape == (batch_size, self.config.n_qubits)
        # Check amplitude normalization (should sum to 1)
        amplitude_sums = torch.sum(quantum_amplitudes, dim=-1)
        assert torch.allclose(amplitude_sums, torch.ones(batch_size), atol=1e-6)


class TestPhotonicQuantumInterface:
    """Test photonic-quantum interface functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = QNPConfig(n_qubits=4, photonic_channels=32, wavelength_bands=8)
        self.interface = PhotonicQuantumInterface(self.config)
    
    def test_interface_initialization(self):
        """Test interface initialization."""
        assert len(self.interface.wavelength_encoders) == self.config.wavelength_bands
        assert hasattr(self.interface, 'coherence_layer')
        assert hasattr(self.interface, 'coupling_strength')
    
    def test_photonic_to_quantum(self):
        """Test photonic to quantum conversion."""
        batch_size = 3
        photonic_signals = torch.randn(batch_size, self.config.photonic_channels, self.config.wavelength_bands)
        
        quantum_states, attention_weights = self.interface.photonic_to_quantum(photonic_signals)
        
        expected_shape = (batch_size, self.config.wavelength_bands, self.config.n_qubits)
        assert quantum_states.shape == expected_shape
        assert attention_weights is not None
    
    def test_quantum_to_photonic(self):
        """Test quantum to photonic conversion."""
        batch_size = 3
        quantum_states = torch.randn(batch_size, self.config.wavelength_bands, self.config.n_qubits)
        
        photonic_signals = self.interface.quantum_to_photonic(quantum_states)
        
        expected_shape = (batch_size, self.config.photonic_channels, self.config.wavelength_bands)
        assert photonic_signals.shape == expected_shape


class TestTriadicFusionLayer:
    """Test triadic fusion layer functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = QNPConfig(n_qubits=4, hidden_dim=64, photonic_channels=32)
        
    def test_fusion_modes(self):
        """Test all fusion modes."""
        batch_size = 3
        quantum_features = torch.randn(batch_size, self.config.n_qubits)
        neuromorphic_features = torch.randn(batch_size, self.config.hidden_dim)
        photonic_features = torch.randn(batch_size, self.config.photonic_channels)
        
        fusion_modes = [FusionMode.EARLY_FUSION, FusionMode.LATE_FUSION,
                       FusionMode.HIERARCHICAL, FusionMode.ADAPTIVE]
        
        for mode in fusion_modes:
            config = QNPConfig(fusion_mode=mode, n_qubits=4, hidden_dim=64, photonic_channels=32)
            fusion_layer = TriadicFusionLayer(config)
            
            fused_output, fusion_info = fusion_layer(
                quantum_features, neuromorphic_features, photonic_features
            )
            
            assert fused_output.shape == (batch_size, config.output_classes)
            assert isinstance(fusion_info, dict)
    
    def test_adaptive_fusion_weights(self):
        """Test adaptive fusion weight computation."""
        config = QNPConfig(fusion_mode=FusionMode.ADAPTIVE, n_qubits=4, hidden_dim=64, photonic_channels=32)
        fusion_layer = TriadicFusionLayer(config)
        
        batch_size = 3
        quantum_features = torch.randn(batch_size, config.n_qubits)
        neuromorphic_features = torch.randn(batch_size, config.hidden_dim)
        photonic_features = torch.randn(batch_size, config.photonic_channels)
        
        fused_output, fusion_info = fusion_layer(
            quantum_features, neuromorphic_features, photonic_features
        )
        
        assert 'adaptive_weights' in fusion_info
        weights = fusion_info['adaptive_weights']
        assert weights.shape == (batch_size, 3)
        # Check weights sum to 1
        weight_sums = torch.sum(weights, dim=-1)
        assert torch.allclose(weight_sums, torch.ones(batch_size), atol=1e-6)


class TestHybridQNPArchitecture:
    """Test complete QNP architecture."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = QNPConfig(n_qubits=4, hidden_dim=32, photonic_channels=16)
        self.model = HybridQNPArchitecture(self.config)
    
    def test_architecture_initialization(self):
        """Test architecture initialization."""
        assert hasattr(self.model, 'quantum_processor')
        assert hasattr(self.model, 'neuromorphic_processor')
        assert hasattr(self.model, 'photonic_processor')
        assert hasattr(self.model, 'qn_bridge')
        assert hasattr(self.model, 'pq_interface')
        assert hasattr(self.model, 'triadic_fusion')
    
    def test_forward_pass(self):
        """Test complete forward pass."""
        batch_size = 3
        input_features = torch.randn(batch_size, self.config.input_dim)
        
        output = self.model(input_features)
        
        # Check required outputs
        assert 'predictions' in output
        assert 'logits' in output
        assert 'quantum_features' in output
        assert 'neuromorphic_features' in output
        assert 'photonic_features' in output
        
        # Check output shapes
        assert output['predictions'].shape == (batch_size, self.config.output_classes)
        assert output['logits'].shape == (batch_size, self.config.output_classes)
        
        # Check predictions are valid probabilities
        predictions = output['predictions']
        assert torch.all(predictions >= 0) and torch.all(predictions <= 1)
        pred_sums = torch.sum(predictions, dim=-1)
        assert torch.allclose(pred_sums, torch.ones(batch_size), atol=1e-6)
    
    def test_entanglement_measurement(self):
        """Test entanglement measurement."""
        self.config.enable_entanglement_measure = True
        model = HybridQNPArchitecture(self.config)
        
        batch_size = 3
        input_features = torch.randn(batch_size, self.config.input_dim)
        
        output = model(input_features)
        
        assert 'analysis_metrics' in output
        if 'qn_entanglement' in output['analysis_metrics']:
            entanglement = output['analysis_metrics']['qn_entanglement']
            assert entanglement.shape == (batch_size,)
            assert torch.all(entanglement >= 0) and torch.all(entanglement <= 1)
    
    def test_coherence_analysis(self):
        """Test coherence analysis."""
        self.config.enable_coherence_analysis = True
        model = HybridQNPArchitecture(self.config)
        
        batch_size = 3
        input_features = torch.randn(batch_size, self.config.input_dim)
        
        output = model(input_features)
        
        assert 'analysis_metrics' in output
        if 'pq_coherence' in output['analysis_metrics']:
            coherence = output['analysis_metrics']['pq_coherence']
            assert coherence.shape == (batch_size,)
            assert torch.all(coherence >= 0) and torch.all(coherence <= 1)
    
    def test_modal_contributions(self):
        """Test modal contribution tracking."""
        self.config.track_modal_contributions = True
        model = HybridQNPArchitecture(self.config)
        
        batch_size = 3
        input_features = torch.randn(batch_size, self.config.input_dim)
        
        output = model(input_features)
        
        assert 'modal_contributions' in output
        contributions = output['modal_contributions']
        
        expected_keys = ['quantum_contribution', 'neuromorphic_contribution', 'photonic_contribution']
        for key in expected_keys:
            assert key in contributions
            assert contributions[key].shape == (batch_size,)
            assert torch.all(contributions[key] >= 0) and torch.all(contributions[key] <= 1)


class TestQNPSentimentAnalyzer:
    """Test QNP sentiment analyzer interface."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = QNPConfig(n_qubits=4, hidden_dim=32)
        self.analyzer = QNPSentimentAnalyzer(self.config)
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        assert self.analyzer.config == self.config
        assert hasattr(self.analyzer, 'model')
        assert self.analyzer.class_labels == ['negative', 'neutral', 'positive']
        assert self.analyzer.trained == False
    
    def test_predict(self):
        """Test prediction functionality."""
        # Generate test features
        test_features = np.random.randn(3, self.config.input_dim)
        
        results = self.analyzer.predict(test_features)
        
        # Check result structure
        assert 'predictions' in results
        assert 'research_analysis' in results
        assert 'architecture_info' in results
        
        # Check predictions
        predictions = results['predictions']
        assert len(predictions) == 3
        
        for pred in predictions:
            assert 'sentiment' in pred
            assert 'confidence' in pred
            assert 'probabilities' in pred
            assert pred['sentiment'] in ['negative', 'neutral', 'positive']
            assert 0 <= pred['confidence'] <= 1
    
    def test_research_analysis(self):
        """Test research analysis compilation."""
        test_features = np.random.randn(2, self.config.input_dim)
        
        results = self.analyzer.predict(test_features)
        research = results['research_analysis']
        
        # Check analysis structure
        assert 'modality_analysis' in research
        assert 'cross_modal_interactions' in research
        assert 'fusion_analysis' in research
        assert 'novel_metrics' in research
        
        # Check modality analysis
        modality = research['modality_analysis']
        expected_keys = ['quantum_activation_strength', 'neuromorphic_activation_strength', 'photonic_activation_strength']
        for key in expected_keys:
            assert key in modality
            assert isinstance(modality[key], float)
    
    def test_benchmark_architectures(self):
        """Test architecture benchmarking."""
        test_features = np.random.randn(5, self.config.input_dim)
        test_labels = np.random.randint(0, 3, size=5)
        
        results = self.analyzer.benchmark_architectures(test_features, test_labels)
        
        # Check all fusion modes were tested
        fusion_modes = ['early', 'late', 'hierarchical', 'adaptive']
        for mode in fusion_modes:
            assert mode in results
            assert 'accuracy' in results[mode]
            assert isinstance(results[mode]['accuracy'], float)
            assert 0 <= results[mode]['accuracy'] <= 1
    
    def test_set_trained(self):
        """Test training status setting."""
        assert self.analyzer.trained == False
        
        self.analyzer.set_trained(True)
        assert self.analyzer.trained == True
        
        self.analyzer.set_trained(False)
        assert self.analyzer.trained == False


class TestQNPFactory:
    """Test QNP factory function."""
    
    def test_create_default_analyzer(self):
        """Test creating analyzer with default config."""
        analyzer = create_qnp_analyzer()
        
        assert isinstance(analyzer, QNPSentimentAnalyzer)
        assert isinstance(analyzer.config, QNPConfig)
    
    def test_create_custom_analyzer(self):
        """Test creating analyzer with custom config."""
        custom_config = {
            'n_qubits': 6,
            'fusion_mode': FusionMode.ADAPTIVE,
            'photonic_channels': 64
        }
        
        analyzer = create_qnp_analyzer(custom_config)
        
        assert analyzer.config.n_qubits == 6
        assert analyzer.config.fusion_mode == FusionMode.ADAPTIVE
        assert analyzer.config.photonic_channels == 64


class TestQNPIntegration:
    """Integration tests for complete QNP system."""
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        # Create analyzer
        analyzer = create_qnp_analyzer()
        
        # Generate test data
        test_features = np.random.randn(5, 768)
        test_labels = np.random.randint(0, 3, size=5)
        
        # Test prediction
        results = analyzer.predict(test_features)
        assert len(results['predictions']) == 5
        
        # Test benchmarking
        benchmark_results = analyzer.benchmark_architectures(test_features, test_labels)
        assert len(benchmark_results) == 4  # 4 fusion modes
        
        # Mark as trained and test again
        analyzer.set_trained(True)
        trained_results = analyzer.predict(test_features)
        assert len(trained_results['predictions']) == 5
    
    def test_different_fusion_modes(self):
        """Test all fusion modes work correctly."""
        test_features = np.random.randn(3, 768)
        
        fusion_modes = [FusionMode.EARLY_FUSION, FusionMode.LATE_FUSION,
                       FusionMode.HIERARCHICAL, FusionMode.ADAPTIVE]
        
        for mode in fusion_modes:
            config = {'fusion_mode': mode}
            analyzer = create_qnp_analyzer(config)
            
            results = analyzer.predict(test_features)
            
            # Check predictions are valid
            assert len(results['predictions']) == 3
            assert results['architecture_info']['fusion_mode'] == mode.value
            
            # Check research analysis
            research = results['research_analysis']
            assert 'modality_analysis' in research
    
    def test_batch_processing(self):
        """Test batch processing capabilities."""
        batch_sizes = [1, 5, 10, 20]
        analyzer = create_qnp_analyzer()
        
        for batch_size in batch_sizes:
            test_features = np.random.randn(batch_size, 768)
            
            results = analyzer.predict(test_features)
            
            assert len(results['predictions']) == batch_size
            
            # Check all predictions are valid
            for pred in results['predictions']:
                assert pred['sentiment'] in ['negative', 'neutral', 'positive']
                assert 0 <= pred['confidence'] <= 1


class TestQNPPerformance:
    """Performance and stress tests for QNP architecture."""
    
    def test_large_batch_processing(self):
        """Test processing large batches."""
        analyzer = create_qnp_analyzer()
        
        # Test with large batch
        large_batch = np.random.randn(100, 768)
        
        import time
        start_time = time.time()
        results = analyzer.predict(large_batch)
        processing_time = time.time() - start_time
        
        assert len(results['predictions']) == 100
        assert processing_time < 60  # Should complete within 60 seconds
    
    def test_memory_efficiency(self):
        """Test memory efficiency."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create multiple analyzers
        analyzers = []
        for _ in range(10):
            analyzer = create_qnp_analyzer()
            analyzers.append(analyzer)
        
        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = current_memory - initial_memory
        
        # Memory increase should be reasonable (less than 500MB for 10 analyzers)
        assert memory_increase < 500
    
    def test_concurrent_processing(self):
        """Test concurrent processing capabilities."""
        import threading
        import time
        
        analyzer = create_qnp_analyzer()
        results = []
        errors = []
        
        def process_batch(batch_id):
            try:
                test_features = np.random.randn(10, 768)
                result = analyzer.predict(test_features)
                results.append(f"batch_{batch_id}")
            except Exception as e:
                errors.append(str(e))
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=process_batch, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        assert len(results) == 5
        assert len(errors) == 0


# Demonstration test
def test_qnp_demonstration():
    """Test the QNP demonstration function."""
    # This should run without errors
    with patch('builtins.print') as mock_print:
        demonstrate_qnp_breakthrough()
        
        # Check that demonstration printed output
        assert mock_print.called
        
        # Verify some expected output
        call_args = [str(call) for call in mock_print.call_args_list]
        demo_output = ' '.join(call_args)
        assert 'Hybrid Quantum-Neuromorphic-Photonic' in demo_output
        assert 'QNP Architecture' in demo_output


# Mark slow tests
pytestmark = pytest.mark.unit

if __name__ == "__main__":
    pytest.main([__file__, "-v"])