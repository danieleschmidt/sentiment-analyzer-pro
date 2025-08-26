"""
Test suite for Quantum-Neuromorphic-Photonic (QNP) Fusion system.
Comprehensive testing of quantum, neuromorphic, and photonic components.
"""

import pytest
import asyncio
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from quantum_neuromorphic_photonic_fusion import (
    QNPFusionEngine,
    QuantumFeatureEncoder,
    NeuromorphicProcessor,
    PhotonicAccelerator,
    QuantumState,
    NeuromorphicSpike,
    PhotonicChannel,
    create_qnp_fusion_engine,
    batch_analyze_sentiment
)

class TestQuantumComponents:
    """Test quantum computing components"""
    
    def test_quantum_state_creation(self):
        """Test quantum state initialization"""
        state = QuantumState(amplitude=complex(0.707, 0.707), phase=np.pi/4)
        assert isinstance(state.amplitude, complex)
        assert state.phase == np.pi/4
        assert state.entanglement_strength == 0.0
        assert state.coherence_time == 1.0
    
    def test_quantum_state_collapse(self):
        """Test quantum state collapse to classical value"""
        state = QuantumState(amplitude=complex(1, 0), phase=0)
        collapsed_value = state.collapse()
        assert isinstance(collapsed_value, (float, np.floating))
        assert 0 <= collapsed_value <= 1
    
    def test_quantum_feature_encoder_initialization(self):
        """Test quantum feature encoder setup"""
        encoder = QuantumFeatureEncoder(num_qubits=8)
        assert encoder.num_qubits == 8
        assert encoder.entanglement_matrix.shape == (8, 8)
        assert isinstance(encoder.quantum_states, list)
    
    def test_quantum_text_encoding(self):
        """Test text encoding to quantum states"""
        encoder = QuantumFeatureEncoder(num_qubits=4)
        text = "test quantum encoding"
        quantum_states = encoder.encode_text(text)
        
        assert len(quantum_states) <= encoder.num_qubits
        assert all(isinstance(state, QuantumState) for state in quantum_states)
        assert all(isinstance(state.amplitude, complex) for state in quantum_states)
    
    def test_quantum_gates_application(self):
        """Test quantum gate operations"""
        encoder = QuantumFeatureEncoder(num_qubits=4)
        initial_states = [QuantumState(amplitude=complex(1, 0), phase=0) for _ in range(3)]
        
        processed_states = encoder.apply_quantum_gates(initial_states)
        
        assert len(processed_states) == len(initial_states)
        assert all(isinstance(state, QuantumState) for state in processed_states)
        # Check that states have been modified
        assert processed_states[0].amplitude != initial_states[0].amplitude

class TestNeuromorphicComponents:
    """Test neuromorphic processing components"""
    
    def test_neuromorphic_spike_creation(self):
        """Test neuromorphic spike initialization"""
        spike = NeuromorphicSpike(timestamp=1.0, neuron_id=5, amplitude=2.0)
        assert spike.timestamp == 1.0
        assert spike.neuron_id == 5
        assert spike.amplitude == 2.0
        assert spike.refractory_period == 0.001
    
    def test_spike_activity_status(self):
        """Test spike activity over time"""
        spike = NeuromorphicSpike(timestamp=1.0, neuron_id=1, amplitude=1.0)
        
        # Should be active immediately after creation
        assert spike.is_active(1.0)
        assert spike.is_active(1.0005)
        
        # Should be inactive after refractory period
        assert not spike.is_active(1.002)
    
    def test_neuromorphic_processor_initialization(self):
        """Test neuromorphic processor setup"""
        processor = NeuromorphicProcessor(num_neurons=64)
        assert processor.num_neurons == 64
        assert processor.neurons.shape == (64,)
        assert processor.synaptic_weights.shape == (64, 64)
        assert processor.membrane_potential.shape == (64,)
    
    def test_quantum_to_spike_conversion(self):
        """Test conversion of quantum states to spikes"""
        processor = NeuromorphicProcessor(num_neurons=16)
        quantum_states = [
            QuantumState(amplitude=complex(2, 0), phase=0) for _ in range(3)
        ]
        
        spikes = processor.process_quantum_states(quantum_states)
        
        assert isinstance(spikes, list)
        assert all(isinstance(spike, NeuromorphicSpike) for spike in spikes)
        # Should generate spikes for high amplitude quantum states
        assert len(spikes) > 0
    
    def test_spike_pattern_computation(self):
        """Test spike pattern analysis"""
        processor = NeuromorphicProcessor(num_neurons=32)
        
        # Add some test spikes
        test_spikes = [
            NeuromorphicSpike(timestamp=1.0, neuron_id=i, amplitude=1.0)
            for i in range(5)
        ]
        processor.spike_history.extend(test_spikes)
        
        pattern = processor.compute_spike_pattern(window_ms=50)
        assert pattern.shape == (32,)
        assert np.any(pattern > 0)  # Should have some activity

class TestPhotonicComponents:
    """Test photonic acceleration components"""
    
    def test_photonic_channel_creation(self):
        """Test photonic channel initialization"""
        channel = PhotonicChannel(wavelength=1550, power=1.0, polarization="horizontal")
        assert channel.wavelength == 1550
        assert channel.power == 1.0
        assert channel.polarization == "horizontal"
        assert channel.coherence_length == 1000.0
    
    def test_channel_interference(self):
        """Test interference between photonic channels"""
        channel1 = PhotonicChannel(wavelength=1550, power=1.0)
        channel2 = PhotonicChannel(wavelength=1551, power=1.0)
        
        interference = channel1.interference_pattern(channel2)
        assert isinstance(interference, (float, np.floating))
        assert interference >= 0  # Interference intensity should be non-negative
    
    def test_photonic_accelerator_initialization(self):
        """Test photonic accelerator setup"""
        accelerator = PhotonicAccelerator(num_channels=16)
        assert accelerator.num_channels == 16
        assert len(accelerator.channels) == 16
        assert accelerator.interference_matrix.shape == (16, 16)
        
        # Check that interference matrix is computed
        assert np.any(accelerator.interference_matrix != 0)
    
    def test_photonic_inference(self):
        """Test photonic-based sentiment inference"""
        accelerator = PhotonicAccelerator(num_channels=8)
        spike_pattern = np.random.random(8) * 2  # Random spike pattern
        
        result = accelerator.photonic_inference(spike_pattern)
        
        # Check result structure
        assert "positive" in result
        assert "negative" in result
        assert "neutral" in result
        assert "confidence" in result
        assert "photonic_coherence" in result
        
        # Check probability constraints
        total_prob = result["positive"] + result["negative"] + result["neutral"]
        assert abs(total_prob - 1.0) < 0.1  # Should approximately sum to 1
        
        # Check value ranges
        assert 0 <= result["positive"] <= 1
        assert 0 <= result["negative"] <= 1
        assert 0 <= result["neutral"] <= 1
        assert 0 <= result["confidence"] <= 1

class TestQNPFusionEngine:
    """Test complete QNP Fusion system"""
    
    def test_engine_initialization(self):
        """Test QNP fusion engine setup"""
        engine = QNPFusionEngine()
        
        assert isinstance(engine.quantum_encoder, QuantumFeatureEncoder)
        assert isinstance(engine.neuromorphic_processor, NeuromorphicProcessor)
        assert isinstance(engine.photonic_accelerator, PhotonicAccelerator)
        assert "total_predictions" in engine.performance_metrics
    
    def test_custom_configuration(self):
        """Test engine with custom configuration"""
        config = {
            "num_qubits": 12,
            "num_neurons": 128,
            "num_photonic_channels": 32
        }
        engine = QNPFusionEngine(config)
        
        assert engine.quantum_encoder.num_qubits == 12
        assert engine.neuromorphic_processor.num_neurons == 128
        assert engine.photonic_accelerator.num_channels == 32
    
    def test_synchronous_sentiment_analysis(self):
        """Test synchronous sentiment analysis"""
        engine = QNPFusionEngine()
        text = "This is a great test of the QNP system!"
        
        result = engine.analyze_sentiment(text)
        
        # Check result structure
        required_keys = ["positive", "negative", "neutral", "confidence", 
                        "processing_time_ms", "qnp_fusion_score"]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"
        
        # Check value types and ranges
        assert isinstance(result["processing_time_ms"], (float, int))
        assert result["processing_time_ms"] > 0
        assert 0 <= result["qnp_fusion_score"] <= 1
    
    @pytest.mark.asyncio
    async def test_asynchronous_sentiment_analysis(self):
        """Test asynchronous sentiment analysis"""
        engine = QNPFusionEngine()
        text = "Asynchronous QNP processing test"
        
        result = await engine.analyze_sentiment_async(text)
        
        assert "positive" in result
        assert "negative" in result
        assert "neutral" in result
        assert "qnp_fusion_score" in result
        assert result["processing_time_ms"] > 0
    
    def test_fallback_analysis(self):
        """Test fallback classical analysis"""
        engine = QNPFusionEngine()
        text = "This is a good test but also bad in some ways"
        
        result = engine.fallback_analysis(text)
        
        assert "positive" in result
        assert "negative" in result
        assert "neutral" in result
        assert "confidence" in result
        assert result.get("fallback") == True
    
    def test_performance_metrics_update(self):
        """Test performance metrics tracking"""
        engine = QNPFusionEngine()
        initial_predictions = engine.performance_metrics["total_predictions"]
        
        # Run analysis to update metrics
        engine.analyze_sentiment("Test text for metrics")
        
        assert engine.performance_metrics["total_predictions"] == initial_predictions + 1
        assert engine.performance_metrics["avg_processing_time"] > 0
    
    def test_performance_report_generation(self):
        """Test comprehensive performance report"""
        engine = QNPFusionEngine()
        engine.analyze_sentiment("Test for report generation")
        
        report = engine.get_performance_report()
        
        assert "qnp_fusion_engine" in report
        assert "quantum_encoder" in report
        assert "neuromorphic_processor" in report
        assert "photonic_accelerator" in report
        
        # Check QNP engine metrics
        qnp_metrics = report["qnp_fusion_engine"]
        assert "total_predictions" in qnp_metrics
        assert "avg_processing_time" in qnp_metrics

class TestFactoryAndBatchFunctions:
    """Test factory functions and batch processing"""
    
    def test_factory_function(self):
        """Test QNP engine factory function"""
        engine = create_qnp_fusion_engine()
        assert isinstance(engine, QNPFusionEngine)
        
        # Test with custom config
        config = {"num_qubits": 6, "num_neurons": 48}
        engine = create_qnp_fusion_engine(config)
        assert engine.quantum_encoder.num_qubits == 6
        assert engine.neuromorphic_processor.num_neurons == 48
    
    @pytest.mark.asyncio
    async def test_batch_processing(self):
        """Test batch sentiment analysis"""
        texts = [
            "Great product!",
            "Terrible experience.",
            "Okay service.",
            "Amazing breakthrough in technology!"
        ]
        
        results = await batch_analyze_sentiment(texts)
        
        assert len(results) == len(texts)
        assert all("positive" in result for result in results)
        assert all("negative" in result for result in results)
        assert all("neutral" in result for result in results)
    
    @pytest.mark.asyncio
    async def test_batch_processing_with_errors(self):
        """Test batch processing error handling"""
        # Include some potentially problematic texts
        texts = [
            "Normal text",
            "",  # Empty text
            "A" * 10000,  # Very long text
            "Special chars: 🚀🔬⚡🌟"  # Unicode
        ]
        
        results = await batch_analyze_sentiment(texts)
        
        assert len(results) == len(texts)
        # All should return valid results (fallback if necessary)
        for result in results:
            assert "positive" in result
            assert "confidence" in result

class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_empty_text_analysis(self):
        """Test analysis of empty text"""
        engine = QNPFusionEngine()
        result = engine.analyze_sentiment("")
        
        assert "positive" in result
        assert "negative" in result
        assert "neutral" in result
        # Should handle gracefully, likely using fallback
    
    def test_very_long_text_analysis(self):
        """Test analysis of very long text"""
        engine = QNPFusionEngine()
        long_text = "This is a test. " * 1000  # Very long text
        
        result = engine.analyze_sentiment(long_text)
        
        assert result["processing_time_ms"] > 0
        assert "qnp_fusion_score" in result
    
    def test_unicode_text_analysis(self):
        """Test analysis of Unicode text"""
        engine = QNPFusionEngine()
        unicode_text = "Great work! 🚀 Amazing results! 🔬⚡🌟"
        
        result = engine.analyze_sentiment(unicode_text)
        
        assert "positive" in result
        assert result["processing_time_ms"] > 0
    
    def test_fusion_score_computation(self):
        """Test QNP fusion score edge cases"""
        engine = QNPFusionEngine()
        
        # Empty quantum states
        score = engine.compute_fusion_score([], [], {"photonic_coherence": 0.5})
        assert score == 0.0
        
        # Normal case
        quantum_states = [QuantumState(amplitude=complex(0.5, 0.5)) for _ in range(3)]
        spikes = [NeuromorphicSpike(timestamp=1.0, neuron_id=i, amplitude=1.0) for i in range(2)]
        photonic_result = {"photonic_coherence": 0.7}
        
        score = engine.compute_fusion_score(quantum_states, spikes, photonic_result)
        assert 0 <= score <= 1

class TestIntegration:
    """Integration tests for complete QNP system"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_processing(self):
        """Test complete end-to-end QNP processing"""
        engine = create_qnp_fusion_engine({
            "num_qubits": 4,
            "num_neurons": 32,
            "num_photonic_channels": 8
        })
        
        test_cases = [
            ("I love this product!", "positive"),
            ("This is terrible quality.", "negative"),
            ("The item is okay, nothing special.", "neutral"),
            ("Excellent breakthrough in quantum computing!", "positive"),
        ]
        
        for text, expected_sentiment in test_cases:
            result = await engine.analyze_sentiment_async(text)
            
            # Check that result contains all expected components
            assert "quantum_states_processed" in result
            assert "neuromorphic_spikes" in result
            assert "qnp_fusion_score" in result
            assert "processing_time_ms" in result
            
            # Verify processing occurred (not just fallback)
            if not result.get("fallback", False):
                assert result["quantum_states_processed"] > 0
                assert result["qnp_fusion_score"] > 0
                
                # Check sentiment alignment (approximate)
                sentiment_scores = {
                    "positive": result["positive"],
                    "negative": result["negative"],
                    "neutral": result["neutral"]
                }
                predicted_sentiment = max(sentiment_scores, key=sentiment_scores.get)
                
                # For clear cases, sentiment should align
                if expected_sentiment != "neutral":
                    assert predicted_sentiment == expected_sentiment or result["confidence"] < 0.3
    
    def test_system_performance_tracking(self):
        """Test system-wide performance tracking"""
        engine = create_qnp_fusion_engine()
        
        # Process multiple texts to build performance history
        test_texts = [
            "Excellent results!",
            "Poor performance.",
            "Average quality.",
            "Outstanding breakthrough!",
            "Disappointing outcome."
        ]
        
        for text in test_texts:
            engine.analyze_sentiment(text)
        
        report = engine.get_performance_report()
        
        # Verify performance metrics are tracked
        assert report["qnp_fusion_engine"]["total_predictions"] == len(test_texts)
        assert report["qnp_fusion_engine"]["avg_processing_time"] > 0
        assert report["qnp_fusion_engine"]["quantum_coherence_avg"] >= 0
        assert report["qnp_fusion_engine"]["neuromorphic_activity_avg"] >= 0

if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])