"""
Comprehensive AGI Integration Tests
Tests for the complete AGI system including all components.
"""

import pytest
import asyncio
import time
import json
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Import AGI components
from src.adaptive_agi_engine import (
    AdaptiveAGIEngine, ReasoningType, CognitiveState, 
    CognitiveMemory, QuantumState, NeuralNetwork
)
from src.enterprise_security_framework import (
    SecurityOrchestrator, ThreatLevel, SecurityEvent
)
from src.intelligent_error_recovery_v2 import (
    IntelligentErrorRecovery, ErrorSeverity, RecoveryStrategy
)
from src.quantum_performance_accelerator import (
    QuantumPerformanceAccelerator, OptimizationLevel, PerformanceMetrics
)


class TestAGIEngine:
    """Test suite for Adaptive AGI Engine."""
    
    @pytest.fixture
    def agi_engine(self):
        """Create AGI engine for testing."""
        return AdaptiveAGIEngine()
    
    @pytest.fixture
    def sample_inputs(self):
        """Sample inputs for testing."""
        return {
            "text": "I love this product! It's amazing.",
            "numerical": [1.0, 2.5, 3.2, 4.1, 5.0],
            "temporal": [time.time() - 60, time.time() - 30, time.time()],
            "categorical": ["positive", "good", "excellent", "positive"]
        }
    
    def test_agi_initialization(self, agi_engine):
        """Test AGI engine initialization."""
        assert agi_engine.state == CognitiveState.IDLE
        assert agi_engine.memory is not None
        assert agi_engine.neural_network is not None
        assert agi_engine.reasoner is not None
        assert agi_engine.multimodal_processor is not None
        assert agi_engine.optimizer is not None
    
    @pytest.mark.asyncio
    async def test_basic_processing(self, agi_engine):
        """Test basic AGI processing."""
        input_text = "Hello, this is a test input"
        result = await agi_engine.process(input_text)
        
        assert "neural_output" in result
        assert "processing_time" in result
        assert "cognitive_state" in result
        assert "quantum_coherence" in result
        assert result["processing_time"] > 0
        assert 0 <= result["neural_output"] <= 1
        assert 0 <= result["quantum_coherence"] <= 1
    
    @pytest.mark.asyncio
    async def test_reasoning_capabilities(self, agi_engine):
        """Test cognitive reasoning capabilities."""
        test_cases = [
            (ReasoningType.DEDUCTIVE, "All humans are mortal. Socrates is human."),
            (ReasoningType.INDUCTIVE, "The sun has risen every day for thousands of years."),
            (ReasoningType.ABDUCTIVE, "The grass is wet in the morning."),
            (ReasoningType.ANALOGICAL, "A heart is like a pump."),
            (ReasoningType.CAUSAL, "Smoking causes lung cancer."),
            (ReasoningType.TEMPORAL, "Stock prices fluctuate over time.")
        ]
        
        for reasoning_type, input_text in test_cases:
            result = await agi_engine.process(
                input_text, 
                reasoning_type=reasoning_type, 
                require_reasoning=True
            )
            
            assert "reasoning" in result
            assert result["reasoning"]["reasoning_type"] == reasoning_type.value
            assert "conclusion" in result["reasoning"]
            assert "confidence" in result["reasoning"]
            assert 0 <= result["reasoning"]["confidence"] <= 1
    
    @pytest.mark.asyncio
    async def test_multimodal_processing(self, agi_engine, sample_inputs):
        """Test multi-modal input processing."""
        result = await agi_engine.process(sample_inputs, multi_modal=True)
        
        assert "multimodal_analysis" in result
        assert "individual_results" in result["multimodal_analysis"]
        assert "fused_result" in result["multimodal_analysis"]
        assert "modalities_used" in result["multimodal_analysis"]
        
        # Check that all modalities were processed
        processed_modalities = result["multimodal_analysis"]["modalities_used"]
        assert "text" in processed_modalities
        assert "numerical" in processed_modalities
    
    def test_memory_systems(self, agi_engine):
        """Test AGI memory systems."""
        # Test episodic memory
        event = {"type": "test", "content": "test memory"}
        agi_engine.memory.store_episodic(event)
        assert len(agi_engine.memory.episodic) > 0
        assert agi_engine.memory.episodic[-1]["type"] == "test"
        
        # Test semantic memory
        agi_engine.memory.semantic["test_concept"] = "test definition"
        assert "test_concept" in agi_engine.memory.semantic
        
        # Test memory retrieval
        similar_memories = agi_engine.memory.retrieve_similar("test")
        assert len(similar_memories) >= 0
    
    def test_neural_network_quantum_integration(self, agi_engine):
        """Test neural network with quantum enhancement."""
        input_vector = np.random.random(100)
        output = agi_engine.neural_network.forward(input_vector.reshape(1, -1))
        
        assert output.shape[0] == 1
        assert output.shape[1] == 1
        assert not np.isnan(output).any()
        
        # Test quantum state evolution
        initial_coherence = agi_engine.neural_network.quantum_state.coherence
        agi_engine.neural_network.quantum_state.evolve()
        evolved_coherence = agi_engine.neural_network.quantum_state.coherence
        
        # Coherence should have changed due to evolution
        assert initial_coherence != evolved_coherence
    
    @pytest.mark.asyncio
    async def test_error_handling(self, agi_engine):
        """Test AGI error handling and recovery."""
        # Test with invalid input
        with pytest.raises(Exception):
            await agi_engine.process(None)
        
        # Test with malformed input
        result = await agi_engine.process("")
        assert "error" in result or "neural_output" in result
    
    def test_performance_optimization(self, agi_engine):
        """Test self-optimization capabilities."""
        system_state = {
            "memory_usage": 0.5,
            "cpu_usage": 0.7,
            "response_time": 2.0,
            "accuracy": 0.8,
            "cache_hit_rate": 0.6
        }
        
        optimization_result = agi_engine.optimizer.optimize(system_state)
        
        assert "optimized_state" in optimization_result
        assert "performance_improvement" in optimization_result
        assert "strategy_used" in optimization_result
    
    def test_system_status(self, agi_engine):
        """Test system status reporting."""
        status = agi_engine.get_system_status()
        
        assert "cognitive_state" in status
        assert "quantum_coherence" in status
        assert "memory_statistics" in status
        assert "performance_statistics" in status
        assert "optimization_status" in status
        assert "multimodal_capabilities" in status
        assert "reasoning_capabilities" in status


class TestSecurityFramework:
    """Test suite for Enterprise Security Framework."""
    
    @pytest.fixture
    def security_framework(self):
        """Create security framework for testing."""
        from src.enterprise_security_framework import create_security_framework
        return create_security_framework()
    
    def test_security_initialization(self, security_framework):
        """Test security framework initialization."""
        assert security_framework.audit_logger is not None
        assert security_framework.behavioral_analyzer is not None
        assert security_framework.rate_limiter is not None
        assert security_framework.encryption_manager is not None
        assert security_framework.compliance_manager is not None
    
    def test_request_validation(self, security_framework):
        """Test request validation capabilities."""
        # Test normal request
        normal_request = {
            "text": "Hello world",
            "user_agent": "pytest/1.0",
            "endpoint": "/test"
        }
        
        validation = security_framework.validate_request(
            normal_request, 
            user_id="test_user",
            source_ip="127.0.0.1"
        )
        
        assert validation["allowed"] == True
        assert validation["security_level"] is not None
    
    def test_threat_detection(self, security_framework):
        """Test threat detection capabilities."""
        # Test SQL injection attempt
        malicious_request = {
            "text": "'; DROP TABLE users; --",
            "user_agent": "hacker/1.0",
            "endpoint": "/vulnerable"
        }
        
        validation = security_framework.validate_request(
            malicious_request,
            user_id="suspicious_user",
            source_ip="192.168.1.100"
        )
        
        # Should detect threats in warnings
        assert len(validation["warnings"]) > 0
    
    def test_rate_limiting(self, security_framework):
        """Test rate limiting functionality."""
        # Test rapid requests
        for i in range(10):
            allowed, info = security_framework.rate_limiter.check_limit("test_user")
            if not allowed:
                assert "reason" in info
                break
    
    def test_encryption_decryption(self, security_framework):
        """Test encryption and decryption."""
        sensitive_data = "This is confidential information"
        
        encrypted = security_framework.encryption_manager.encrypt_sensitive_data(
            sensitive_data, "test_purpose"
        )
        
        assert "encrypted_data" in encrypted
        assert "purpose" in encrypted
        
        decrypted = security_framework.encryption_manager.decrypt_sensitive_data(encrypted)
        assert decrypted == sensitive_data
    
    def test_compliance_management(self, security_framework):
        """Test compliance management features."""
        # Test data processing record
        record_id = security_framework.compliance_manager.record_data_processing(
            "test_user", "personal_data", "sentiment_analysis", "consent"
        )
        
        assert record_id is not None
        assert len(record_id) > 0
        
        # Test consent recording
        consent = security_framework.compliance_manager.record_consent(
            "test_user", "data_processing", True
        )
        
        assert consent["user_id"] == "test_user"
        assert consent["granted"] == True
    
    def test_security_incident_handling(self, security_framework):
        """Test security incident handling."""
        event = security_framework.handle_security_incident(
            "test_incident",
            ThreatLevel.MEDIUM,
            source_ip="192.168.1.100",
            user_id="test_user"
        )
        
        assert isinstance(event, SecurityEvent)
        assert event.event_type == "test_incident"
        assert event.severity == ThreatLevel.MEDIUM
    
    def test_security_dashboard(self, security_framework):
        """Test security dashboard data."""
        dashboard = security_framework.get_security_dashboard()
        
        assert "threat_summary" in dashboard
        assert "active_threats" in dashboard
        assert "blocked_ips" in dashboard
        assert "compliance_status" in dashboard
        assert "encryption_status" in dashboard


class TestErrorRecovery:
    """Test suite for Intelligent Error Recovery System."""
    
    @pytest.fixture
    def error_recovery(self):
        """Create error recovery system for testing."""
        from src.intelligent_error_recovery_v2 import create_error_recovery_system
        return create_error_recovery_system()
    
    def test_error_recovery_initialization(self, error_recovery):
        """Test error recovery system initialization."""
        assert error_recovery.system_monitor is not None
        assert error_recovery.error_patterns is not None
        assert error_recovery.recovery_plans is not None
    
    def test_error_classification(self, error_recovery):
        """Test error classification capabilities."""
        # Test different error types
        memory_error = MemoryError("Out of memory")
        value_error = ValueError("Invalid value")
        connection_error = ConnectionError("Connection failed")
        
        event1 = error_recovery.handle_error(memory_error)
        event2 = error_recovery.handle_error(value_error)
        event3 = error_recovery.handle_error(connection_error)
        
        assert event1.severity == ErrorSeverity.CRITICAL
        assert event2.severity == ErrorSeverity.LOW
        assert event3.severity == ErrorSeverity.MEDIUM
    
    def test_recovery_strategies(self, error_recovery):
        """Test error recovery strategies."""
        test_error = Exception("Test error")
        
        # Test with auto-recovery enabled
        event = error_recovery.handle_error(test_error, auto_recover=True)
        
        assert event.recovery_attempted == True
        assert "recovery_strategy" in event.__dict__
    
    def test_system_monitoring(self, error_recovery):
        """Test system health monitoring."""
        # Start monitoring
        error_recovery.system_monitor.start_monitoring(interval=0.1)
        
        # Let it collect some metrics
        time.sleep(0.2)
        
        health_report = error_recovery.system_monitor.get_health_report()
        
        assert "current_state" in health_report
        assert "current_metrics" in health_report
        assert "monitoring_active" in health_report
        
        # Stop monitoring
        error_recovery.system_monitor.stop_monitoring()
    
    def test_circuit_breaker(self, error_recovery):
        """Test circuit breaker functionality."""
        from src.intelligent_error_recovery_v2 import CircuitBreaker
        
        circuit_breaker = CircuitBreaker(failure_threshold=3)
        
        def failing_function():
            raise Exception("Simulated failure")
        
        # Test circuit breaker
        for i in range(5):
            try:
                circuit_breaker.call(failing_function)
            except Exception:
                pass
        
        metrics = circuit_breaker.get_metrics()
        assert metrics["state"] in ["CLOSED", "OPEN", "HALF_OPEN"]
        assert metrics["failure_count"] >= 0
    
    def test_adaptive_retry(self, error_recovery):
        """Test adaptive retry mechanism."""
        from src.intelligent_error_recovery_v2 import AdaptiveRetry
        
        retry_handler = AdaptiveRetry(max_retries=3)
        
        call_count = 0
        def intermittent_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary failure")
            return "success"
        
        result = retry_handler.execute(
            intermittent_function,
            (ValueError,),
            None
        )
        
        assert result == "success"
        assert call_count == 3
    
    def test_recovery_dashboard(self, error_recovery):
        """Test recovery system dashboard."""
        dashboard = error_recovery.get_recovery_dashboard()
        
        assert "system_health" in dashboard
        assert "error_statistics" in dashboard
        assert "circuit_breaker_status" in dashboard
        assert "recovery_plans" in dashboard


class TestQuantumPerformanceAccelerator:
    """Test suite for Quantum Performance Accelerator."""
    
    @pytest.fixture
    def quantum_accelerator(self):
        """Create quantum accelerator for testing."""
        from src.quantum_performance_accelerator import create_quantum_accelerator
        return create_quantum_accelerator()
    
    def test_quantum_accelerator_initialization(self, quantum_accelerator):
        """Test quantum accelerator initialization."""
        assert quantum_accelerator.quantum_optimizer is not None
        assert quantum_accelerator.neural_compressor is not None
        assert quantum_accelerator.resource_manager is not None
    
    def test_function_optimization(self, quantum_accelerator):
        """Test function optimization capabilities."""
        @quantum_accelerator.optimize_function
        def test_function(x, y):
            time.sleep(0.01)  # Simulate work
            return x + y
        
        result = test_function(5, 3)
        assert result == 8
    
    def test_batch_processing_optimization(self, quantum_accelerator):
        """Test batch processing optimization."""
        def simple_processor(item):
            return item * 2
        
        data_batch = list(range(100))
        results = quantum_accelerator.optimize_batch_processing(
            data_batch, simple_processor, batch_size=10
        )
        
        assert len(results) == 100
        assert results[0] == 0
        assert results[50] == 100
    
    def test_quantum_hyperparameter_optimization(self, quantum_accelerator):
        """Test quantum hyperparameter optimization."""
        def model_factory(learning_rate, batch_size):
            # Mock model that returns score based on parameters
            return {"lr": learning_rate, "bs": batch_size}
        
        def evaluation_metric(model):
            # Mock evaluation that prefers certain parameter ranges
            return 1.0 - abs(model["lr"] - 0.01) - abs(model["bs"] - 32) / 100
        
        parameter_space = {
            "learning_rate": (0.001, 0.1),
            "batch_size": (16, 64)
        }
        
        result = quantum_accelerator.quantum_hyperparameter_optimization(
            model_factory, parameter_space, evaluation_metric, iterations=10
        )
        
        assert "best_parameters" in result
        assert "best_score" in result
        assert "quantum_coherence" in result
        assert 0.001 <= result["best_parameters"]["learning_rate"] <= 0.1
        assert 16 <= result["best_parameters"]["batch_size"] <= 64
    
    def test_resource_management(self, quantum_accelerator):
        """Test adaptive resource management."""
        from src.quantum_performance_accelerator import ResourceType
        
        task_requirements = {
            ResourceType.CPU: 2.0,
            ResourceType.MEMORY: 1000.0 * 1024 * 1024  # 1GB
        }
        
        allocation = quantum_accelerator.resource_manager.allocate_resources(
            task_requirements
        )
        
        if allocation["success"]:
            assert "allocation_id" in allocation
            assert "allocated_resources" in allocation
            
            # Release resources
            release_result = quantum_accelerator.resource_manager.release_resources(
                allocation["allocation_id"]
            )
            assert release_result["success"] == True
    
    def test_neural_network_compression(self, quantum_accelerator):
        """Test neural network compression."""
        # Create a mock neural network
        class MockNetwork:
            def __init__(self):
                self.weights = [
                    np.random.random((10, 5)),
                    np.random.random((5, 1))
                ]
        
        mock_network = MockNetwork()
        
        compression_result = quantum_accelerator.neural_compressor.compress_network(
            mock_network, target_compression=0.5
        )
        
        assert "compressed_network" in compression_result
        assert "compression_ratio" in compression_result
        assert "techniques_used" in compression_result
    
    def test_quantum_optimization_algorithms(self, quantum_accelerator):
        """Test quantum optimization algorithms."""
        def simple_cost_function(x):
            # Simple quadratic function with minimum at x=0.5
            return (x - 0.5) ** 2
        
        initial_state = np.array([0.0])
        temperature_schedule = [1.0, 0.5, 0.1]
        
        best_state, best_cost = quantum_accelerator.quantum_optimizer.quantum_annealing(
            simple_cost_function, initial_state, temperature_schedule, iterations=10
        )
        
        assert len(best_state) == 1
        assert best_cost >= 0
        assert abs(best_state[0] - 0.5) < 0.5  # Should be close to optimal
    
    def test_performance_monitoring(self, quantum_accelerator):
        """Test performance monitoring capabilities."""
        with quantum_accelerator.performance_monitor("test_operation"):
            time.sleep(0.01)  # Simulate work
        
        # Check that performance was recorded
        assert len(quantum_accelerator.performance_history) > 0
        last_entry = quantum_accelerator.performance_history[-1]
        assert last_entry["operation"] == "test_operation"
        assert "metrics" in last_entry
    
    def test_performance_dashboard(self, quantum_accelerator):
        """Test performance dashboard."""
        dashboard = quantum_accelerator.get_performance_dashboard()
        
        assert "performance_history" in dashboard
        assert "optimization_results" in dashboard
        assert "cache_statistics" in dashboard
        assert "resource_status" in dashboard
        assert "quantum_status" in dashboard


class TestSystemIntegration:
    """Test suite for complete system integration."""
    
    @pytest.fixture
    def integrated_system(self):
        """Create fully integrated system for testing."""
        from src.adaptive_agi_engine import create_agi_engine
        from src.enterprise_security_framework import create_security_framework
        from src.intelligent_error_recovery_v2 import create_error_recovery_system
        from src.quantum_performance_accelerator import create_quantum_accelerator
        
        return {
            "agi": create_agi_engine(),
            "security": create_security_framework(),
            "recovery": create_error_recovery_system(),
            "quantum": create_quantum_accelerator()
        }
    
    @pytest.mark.asyncio
    async def test_full_system_integration(self, integrated_system):
        """Test complete system working together."""
        agi = integrated_system["agi"]
        security = integrated_system["security"]
        
        # Test that systems can work together
        test_input = "This is a comprehensive integration test"
        
        # Security validation
        validation = security.validate_request(
            {"text": test_input}, 
            user_id="test_user",
            source_ip="127.0.0.1"
        )
        
        if validation["allowed"]:
            # AGI processing
            result = await agi.process(test_input)
            assert "neural_output" in result
            assert "processing_time" in result
        
        # Check system status
        for system_name, system in integrated_system.items():
            if hasattr(system, 'get_system_status'):
                status = system.get_system_status()
                assert status is not None
            elif hasattr(system, 'get_security_dashboard'):
                dashboard = system.get_security_dashboard()
                assert dashboard is not None
            elif hasattr(system, 'get_recovery_dashboard'):
                dashboard = system.get_recovery_dashboard()
                assert dashboard is not None
            elif hasattr(system, 'get_performance_dashboard'):
                dashboard = system.get_performance_dashboard()
                assert dashboard is not None
    
    def test_error_recovery_integration(self, integrated_system):
        """Test error recovery across systems."""
        recovery = integrated_system["recovery"]
        
        # Simulate error in AGI system
        test_error = Exception("Integration test error")
        event = recovery.handle_error(
            test_error, 
            context={"system": "agi", "operation": "processing"}
        )
        
        assert event.error_type == "Exception"
        assert "system" in event.context
    
    def test_security_integration(self, integrated_system):
        """Test security across all systems."""
        security = integrated_system["security"]
        
        # Test security incident
        event = security.handle_security_incident(
            "integration_test_incident",
            ThreatLevel.LOW,
            source_ip="127.0.0.1"
        )
        
        assert event.event_type == "integration_test_incident"
        
        # Test compliance
        record_id = security.compliance_manager.record_data_processing(
            "integration_user", "test_data", "integration_test", "consent"
        )
        
        assert record_id is not None
    
    def test_performance_optimization_integration(self, integrated_system):
        """Test performance optimization across systems."""
        quantum = integrated_system["quantum"]
        
        # Test resource allocation for integrated operation
        from src.quantum_performance_accelerator import ResourceType
        
        requirements = {
            ResourceType.CPU: 1.0,
            ResourceType.MEMORY: 500 * 1024 * 1024  # 500MB
        }
        
        allocation = quantum.resource_manager.allocate_resources(requirements)
        
        if allocation["success"]:
            # Simulate integrated work
            time.sleep(0.01)
            
            # Release resources
            release_result = quantum.resource_manager.release_resources(
                allocation["allocation_id"]
            )
            assert release_result["success"] == True


# Performance benchmarks
class TestPerformanceBenchmarks:
    """Performance benchmark tests."""
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_agi_processing_performance(self):
        """Benchmark AGI processing performance."""
        from src.adaptive_agi_engine import create_agi_engine
        
        agi = create_agi_engine()
        
        # Benchmark basic processing
        start_time = time.time()
        iterations = 10
        
        for i in range(iterations):
            result = await agi.process(f"Test input {i}")
            assert "neural_output" in result
        
        total_time = time.time() - start_time
        avg_time = total_time / iterations
        
        # Performance assertion (should process in reasonable time)
        assert avg_time < 1.0, f"Average processing time {avg_time:.3f}s exceeds 1.0s threshold"
        
        print(f"AGI Processing Performance: {avg_time:.3f}s average per request")
    
    @pytest.mark.performance
    def test_security_validation_performance(self):
        """Benchmark security validation performance."""
        from src.enterprise_security_framework import create_security_framework
        
        security = create_security_framework()
        
        # Benchmark validation
        start_time = time.time()
        iterations = 100
        
        for i in range(iterations):
            validation = security.validate_request(
                {"text": f"Test request {i}"},
                user_id=f"user_{i}",
                source_ip="127.0.0.1"
            )
            assert validation["allowed"] is not None
        
        total_time = time.time() - start_time
        avg_time = total_time / iterations
        
        # Performance assertion
        assert avg_time < 0.1, f"Average validation time {avg_time:.3f}s exceeds 0.1s threshold"
        
        print(f"Security Validation Performance: {avg_time:.3f}s average per request")
    
    @pytest.mark.performance
    def test_quantum_optimization_performance(self):
        """Benchmark quantum optimization performance."""
        from src.quantum_performance_accelerator import create_quantum_accelerator
        
        quantum = create_quantum_accelerator()
        
        # Benchmark function optimization
        call_count = 0
        
        @quantum.optimize_function
        def benchmark_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2
        
        start_time = time.time()
        iterations = 50
        
        for i in range(iterations):
            result = benchmark_function(i)
            assert result == i * 2
        
        total_time = time.time() - start_time
        avg_time = total_time / iterations
        
        # Performance assertion
        assert avg_time < 0.05, f"Average optimization time {avg_time:.3f}s exceeds 0.05s threshold"
        
        print(f"Quantum Optimization Performance: {avg_time:.3f}s average per call")


# Stress tests
class TestStressTests:
    """Stress testing for system limits."""
    
    @pytest.mark.stress
    @pytest.mark.asyncio
    async def test_agi_concurrent_processing(self):
        """Stress test AGI with concurrent requests."""
        from src.adaptive_agi_engine import create_agi_engine
        import asyncio
        
        agi = create_agi_engine()
        
        async def process_request(request_id):
            result = await agi.process(f"Concurrent request {request_id}")
            return result["neural_output"]
        
        # Create concurrent tasks
        tasks = [process_request(i) for i in range(20)]
        
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        # Check results
        successful_results = [r for r in results if isinstance(r, float)]
        failed_results = [r for r in results if isinstance(r, Exception)]
        
        print(f"Concurrent Processing: {len(successful_results)}/{len(tasks)} successful in {total_time:.2f}s")
        
        # Should handle most concurrent requests successfully
        assert len(successful_results) >= len(tasks) * 0.8, "Too many concurrent requests failed"
    
    @pytest.mark.stress
    def test_security_rate_limiting_stress(self):
        """Stress test security rate limiting."""
        from src.enterprise_security_framework import create_security_framework
        
        security = create_security_framework()
        
        # Rapid requests from same source
        blocked_count = 0
        allowed_count = 0
        
        for i in range(200):
            validation = security.validate_request(
                {"text": f"Rapid request {i}"},
                user_id="stress_user",
                source_ip="192.168.1.100"
            )
            
            if validation["allowed"]:
                allowed_count += 1
            else:
                blocked_count += 1
        
        print(f"Rate Limiting Stress: {allowed_count} allowed, {blocked_count} blocked")
        
        # Should eventually start blocking requests
        assert blocked_count > 0, "Rate limiting should block some requests under stress"
    
    @pytest.mark.stress
    def test_memory_usage_under_load(self):
        """Test memory usage under sustained load."""
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create multiple system instances
        systems = []
        for i in range(5):
            from src.adaptive_agi_engine import create_agi_engine
            systems.append(create_agi_engine())
        
        # Force garbage collection
        gc.collect()
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory
        
        print(f"Memory Usage: {initial_memory:.1f}MB -> {peak_memory:.1f}MB (+{memory_increase:.1f}MB)")
        
        # Should not use excessive memory
        assert memory_increase < 500, f"Memory increase {memory_increase:.1f}MB exceeds 500MB threshold"
        
        # Cleanup
        del systems
        gc.collect()


if __name__ == "__main__":
    # Run specific test categories
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "performance":
            pytest.main(["-v", "-m", "performance", __file__])
        elif sys.argv[1] == "stress":
            pytest.main(["-v", "-m", "stress", __file__])
        else:
            pytest.main(["-v", __file__])
    else:
        pytest.main(["-v", __file__])