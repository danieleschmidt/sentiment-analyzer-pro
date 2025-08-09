"""Comprehensive tests for advanced features and integrations."""

import pytest
import asyncio
import threading
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

# Import our new advanced modules
from src.i18n import t, set_language, get_supported_languages, SupportedLanguages
from src.compliance import (
    get_compliance_manager, ComplianceRegion, DataProcessingPurpose,
    ConsentRecord, ComplianceManager
)
from src.multi_region_deployment import (
    get_region_manager, get_load_balancer, Region, RegionConfig, route_request
)
from src.advanced_error_handling import (
    robust_operation, ErrorCategory, ErrorSeverity, CircuitBreaker,
    RetryStrategy, get_error_manager, error_boundary
)
from src.security_hardening import (
    get_threat_detector, get_security_middleware, InputSanitizer,
    ThreatLevel, AttackType
)
from src.health_monitoring import (
    get_health_monitor, HealthChecker, ComponentType, HealthStatus,
    SystemHealthMonitor, setup_default_health_checkers
)
from src.advanced_caching import (
    get_cache_manager, MemoryCache, CachePolicy, cache_result
)
from src.auto_scaling_advanced import (
    get_advanced_auto_scaler, ScalingMetrics, ScalingRule, ResourceType
)
from src.quantum_research_framework import (
    get_research_manager, QuantumInspiredProcessor, ResearchHypothesis,
    setup_quantum_sentiment_experiment
)

class TestInternationalization:
    """Test internationalization features."""
    
    def test_supported_languages(self):
        """Test getting supported languages."""
        languages = get_supported_languages()
        assert isinstance(languages, list)
        assert len(languages) > 0
        assert 'en' in languages
        assert 'es' in languages
    
    def test_language_switching(self):
        """Test language switching functionality."""
        # Test valid language
        set_language('es')
        message = t('processing')
        assert isinstance(message, str)
        assert message != 'processing'  # Should be translated
        
        # Test invalid language (should fall back to default)
        set_language('invalid')
        message = t('processing')
        assert isinstance(message, str)
    
    def test_translation_keys(self):
        """Test various translation keys."""
        set_language('en')
        
        keys_to_test = [
            'model_training',
            'positive_sentiment', 
            'error_occurred',
            'validation_error'
        ]
        
        for key in keys_to_test:
            translation = t(key)
            assert isinstance(translation, str)
            assert len(translation) > 0
    
    def test_enum_values(self):
        """Test supported languages enum."""
        assert SupportedLanguages.EN.value == 'en'
        assert SupportedLanguages.ES.value == 'es'
        assert SupportedLanguages.FR.value == 'fr'

class TestCompliance:
    """Test compliance and data protection features."""
    
    def test_compliance_manager_initialization(self):
        """Test compliance manager initialization."""
        manager = get_compliance_manager()
        assert isinstance(manager, ComplianceManager)
        assert manager.region == ComplianceRegion.GLOBAL
    
    def test_consent_recording(self):
        """Test user consent recording."""
        manager = get_compliance_manager()
        
        consent = manager.record_consent(
            user_id="test_user_123",
            purpose=DataProcessingPurpose.SENTIMENT_ANALYSIS,
            granted=True
        )
        
        assert isinstance(consent, ConsentRecord)
        assert consent.user_id == "test_user_123"
        assert consent.purpose == DataProcessingPurpose.SENTIMENT_ANALYSIS
        assert consent.granted is True
        assert isinstance(consent.timestamp, datetime)
    
    def test_consent_checking(self):
        """Test consent validation."""
        manager = get_compliance_manager()
        
        # Record consent
        manager.record_consent(
            user_id="test_user_456",
            purpose=DataProcessingPurpose.MODEL_TRAINING,
            granted=True
        )
        
        # Check consent
        has_consent = manager.check_consent(
            user_id="test_user_456",
            purpose=DataProcessingPurpose.MODEL_TRAINING
        )
        assert has_consent is True
        
        # Check different purpose (should be False)
        has_consent = manager.check_consent(
            user_id="test_user_456",
            purpose=DataProcessingPurpose.PERFORMANCE_ANALYTICS
        )
        assert has_consent is False
    
    def test_data_processing(self):
        """Test data processing with compliance."""
        manager = get_compliance_manager()
        
        # Record consent first
        manager.record_consent(
            user_id="test_user_789",
            purpose=DataProcessingPurpose.SENTIMENT_ANALYSIS,
            granted=True
        )
        
        # Process data
        processing_id = manager.process_data(
            user_id="test_user_789",
            data_type="text",
            purpose=DataProcessingPurpose.SENTIMENT_ANALYSIS
        )
        
        assert processing_id is not None
        assert isinstance(processing_id, str)
    
    def test_data_deletion(self):
        """Test user data deletion (right to be forgotten)."""
        manager = get_compliance_manager()
        
        # Add some data
        manager.record_consent(
            user_id="test_delete_user",
            purpose=DataProcessingPurpose.SENTIMENT_ANALYSIS,
            granted=True
        )
        
        # Delete user data
        deletion_success = manager.handle_deletion_request("test_delete_user")
        assert deletion_success is True
        
        # Verify data is gone
        user_data = manager.get_user_data("test_delete_user")
        assert len(user_data['consents']) == 0

class TestMultiRegionDeployment:
    """Test multi-region deployment features."""
    
    def test_region_manager_initialization(self):
        """Test region manager setup."""
        manager = get_region_manager()
        assert len(manager.regions) > 0
        assert Region.US_EAST_1 in manager.regions
    
    def test_optimal_region_selection(self):
        """Test optimal region selection."""
        manager = get_region_manager()
        
        # Test without user location
        optimal_region = manager.get_optimal_region()
        assert isinstance(optimal_region, Region)
        
        # Test with user location (US East Coast)
        user_location = {"latitude": 40.7128, "longitude": -74.0060}
        optimal_region = manager.get_optimal_region(user_location)
        assert isinstance(optimal_region, Region)
    
    def test_health_checks(self):
        """Test region health checking."""
        manager = get_region_manager()
        
        # Health check individual region (will fail in test env, but should not crash)
        result = manager.health_check_region(Region.US_EAST_1)
        assert isinstance(result, bool)
        
        # Health check all regions
        results = manager.health_check_all_regions()
        assert isinstance(results, dict)
        assert len(results) > 0
    
    def test_load_balancer(self):
        """Test global load balancer."""
        load_balancer = get_load_balancer()
        
        request_data = {"text": "test sentiment"}
        routing_info = load_balancer.route_request(request_data)
        
        assert 'region' in routing_info
        assert 'endpoint' in routing_info
        assert 'request_id' in routing_info
    
    def test_route_request_function(self):
        """Test global request routing function."""
        routing_info = route_request(
            {"text": "test"},
            {"latitude": 37.7749, "longitude": -122.4194}  # San Francisco
        )
        
        assert isinstance(routing_info, dict)
        assert 'region' in routing_info

class TestAdvancedErrorHandling:
    """Test advanced error handling features."""
    
    def test_circuit_breaker(self):
        """Test circuit breaker functionality."""
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=1.0)
        
        def failing_function():
            raise Exception("Test failure")
        
        def working_function():
            return "success"
        
        # Test failures trigger circuit breaker
        with pytest.raises(Exception):
            cb.call(failing_function)
        
        with pytest.raises(Exception):
            cb.call(failing_function)
        
        # Circuit should now be open
        with pytest.raises(Exception, match="Circuit breaker"):
            cb.call(working_function)
        
        # Test stats
        stats = cb.get_stats()
        assert stats['failure_count'] >= 2
        assert stats['state'] == 'open'
    
    def test_retry_strategy(self):
        """Test retry strategy."""
        retry = RetryStrategy(max_attempts=3, base_delay=0.1)
        
        attempt_count = 0
        def flaky_function():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise Exception("Temporary failure")
            return "success"
        
        result = retry.execute(flaky_function)
        assert result == "success"
        assert attempt_count == 3
    
    def test_robust_operation_decorator(self):
        """Test robust operation decorator."""
        @robust_operation(
            category=ErrorCategory.PROCESSING,
            severity=ErrorSeverity.MEDIUM
        )
        def test_function(should_fail=False):
            if should_fail:
                raise ValueError("Test error")
            return "success"
        
        # Test successful execution
        result = test_function(should_fail=False)
        assert result == "success"
        
        # Test error handling
        with pytest.raises(ValueError):
            test_function(should_fail=True)
    
    def test_error_boundary(self):
        """Test error boundary context manager."""
        with pytest.raises(ValueError):
            with error_boundary(category=ErrorCategory.VALIDATION):
                raise ValueError("Test validation error")
    
    def test_error_manager(self):
        """Test error recovery manager."""
        manager = get_error_manager()
        
        # Test circuit breaker creation
        cb = manager.get_circuit_breaker("test_cb")
        assert isinstance(cb, CircuitBreaker)
        
        # Test stats
        stats = manager.get_error_statistics()
        assert isinstance(stats, dict)

class TestSecurityHardening:
    """Test security hardening features."""
    
    def test_threat_detector(self):
        """Test threat detection."""
        detector = get_threat_detector()
        
        # Test clean request
        threat_level, threats, should_block = detector.analyze_request(
            ip="192.168.1.1",
            user_agent="Mozilla/5.0",
            path="/predict",
            payload={"text": "I love this product"}
        )
        
        assert isinstance(threat_level, ThreatLevel)
        assert isinstance(threats, list)
        assert isinstance(should_block, bool)
    
    def test_malicious_request_detection(self):
        """Test detection of malicious requests."""
        detector = get_threat_detector()
        
        # Test SQL injection attempt
        threat_level, threats, should_block = detector.analyze_request(
            ip="192.168.1.100",
            user_agent="sqlmap/1.0",
            path="/predict",
            payload={"text": "'; DROP TABLE users; --"}
        )
        
        assert len(threats) > 0
        assert should_block is True
    
    def test_input_sanitizer(self):
        """Test input sanitization."""
        # Test malicious input
        sanitized, warnings = InputSanitizer.sanitize_input("Hello <script>alert('xss')</script>")
        assert "<script>" not in sanitized
        assert len(warnings) > 0
        
        # Test clean input
        sanitized, warnings = InputSanitizer.sanitize_input("Hello world")
        assert sanitized == "Hello world"
        assert len(warnings) == 0
    
    def test_json_payload_validation(self):
        """Test JSON payload validation."""
        malicious_payload = {
            "text": "Hello <script>alert('xss')</script>",
            "user': 'admin'": "value"
        }
        
        sanitized, warnings = InputSanitizer.validate_json_payload(malicious_payload)
        assert isinstance(sanitized, dict)
        assert len(warnings) > 0
    
    def test_security_middleware(self):
        """Test security middleware."""
        middleware = get_security_middleware()
        
        # Test rate limiting
        for i in range(5):
            allowed, response = middleware.process_request(
                ip="test.ip",
                user_agent="test",
                path="/test",
                method="GET",
                headers={},
                payload=None
            )
            assert allowed is True
        
        # Test CSRF token generation and validation
        token = middleware.generate_csrf_token("session123")
        assert isinstance(token, str)
        assert len(token) > 0
        
        is_valid = middleware.validate_csrf_token(token)
        assert is_valid is True

class TestHealthMonitoring:
    """Test health monitoring system."""
    
    def test_health_monitor_initialization(self):
        """Test health monitor setup."""
        monitor = get_health_monitor()
        assert not monitor.is_running
    
    def test_health_checker_creation(self):
        """Test creating health checker."""
        def mock_health_check():
            return {"status": "healthy", "response_time": 50}
        
        checker = HealthChecker(
            name="test_service",
            component_type=ComponentType.API_ENDPOINT,
            check_function=mock_health_check,
            interval=1
        )
        
        assert checker.name == "test_service"
        assert checker.component_type == ComponentType.API_ENDPOINT
        assert checker.interval == 1
    
    def test_system_health_checks(self):
        """Test system health checking functions."""
        # Test system resources check
        resources = SystemHealthMonitor.check_system_resources()
        assert isinstance(resources, dict)
        assert 'cpu_percent' in resources or 'error' in resources
        
        # Test network connectivity check
        network = SystemHealthMonitor.check_network_connectivity()
        assert isinstance(network, dict)
    
    def test_setup_default_checkers(self):
        """Test setting up default health checkers."""
        setup_default_health_checkers()
        
        monitor = get_health_monitor()
        assert len(monitor.health_checkers) > 0

class TestAdvancedCaching:
    """Test advanced caching system."""
    
    def test_memory_cache(self):
        """Test memory cache functionality."""
        cache = MemoryCache(max_size=10, policy=CachePolicy.LRU)
        
        # Test basic operations
        cache.set("key1", "value1")
        value = cache.get("key1")
        assert value == "value1"
        
        # Test miss
        value = cache.get("nonexistent")
        assert value is None
        
        # Test deletion
        success = cache.delete("key1")
        assert success is True
        
        value = cache.get("key1")
        assert value is None
    
    def test_cache_policies(self):
        """Test different cache eviction policies."""
        # Test LRU
        lru_cache = MemoryCache(max_size=2, policy=CachePolicy.LRU)
        lru_cache.set("a", 1)
        lru_cache.set("b", 2)
        lru_cache.set("c", 3)  # Should evict 'a'
        
        assert lru_cache.get("a") is None
        assert lru_cache.get("b") == 2
        assert lru_cache.get("c") == 3
    
    def test_cache_manager(self):
        """Test cache manager functionality."""
        manager = get_cache_manager()
        
        # Test basic operations
        manager.set("test_key", "test_value", ttl=60)
        value = manager.get("test_key")
        assert value == "test_value"
        
        # Test with tags
        manager.set("tagged_key", "tagged_value", tags=["test", "demo"])
        
        # Test tag-based deletion
        deleted_count = manager.delete_by_tag("test")
        assert deleted_count >= 1
    
    def test_cache_decorator(self):
        """Test cache result decorator."""
        call_count = 0
        
        @cache_result(ttl=60)
        def expensive_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2
        
        # First call
        result1 = expensive_function(5)
        assert result1 == 10
        assert call_count == 1
        
        # Second call (should be cached)
        result2 = expensive_function(5)
        assert result2 == 10
        assert call_count == 1  # Should not increment
    
    def test_cache_metrics(self):
        """Test cache metrics tracking."""
        cache = MemoryCache(max_size=10)
        
        # Generate some cache activity
        cache.set("k1", "v1")
        cache.get("k1")  # Hit
        cache.get("k2")  # Miss
        
        metrics = cache.get_metrics()
        assert metrics.hits >= 1
        assert metrics.misses >= 1
        assert metrics.hit_rate >= 0

class TestAutoScaling:
    """Test advanced auto-scaling features."""
    
    def test_auto_scaler_initialization(self):
        """Test auto-scaler setup."""
        scaler = get_advanced_auto_scaler()
        assert not scaler.is_running
    
    def test_scaling_metrics(self):
        """Test scaling metrics creation."""
        metrics = ScalingMetrics(
            timestamp=datetime.now(),
            cpu_usage=75.0,
            memory_usage=60.0,
            request_rate=25.0,
            response_time_ms=100.0,
            queue_depth=5,
            error_rate=1.0
        )
        
        assert metrics.cpu_usage == 75.0
        assert isinstance(metrics.to_dict(), dict)
    
    def test_scaling_rule(self):
        """Test scaling rule creation."""
        rule = ScalingRule(
            name="cpu_scale",
            metric="cpu_usage",
            threshold_up=80.0,
            threshold_down=30.0,
            scale_up_by=1,
            scale_down_by=1
        )
        
        assert rule.name == "cpu_scale"
        assert rule.enabled is True
    
    def test_metrics_recording(self):
        """Test recording metrics."""
        scaler = get_advanced_auto_scaler()
        
        metrics = ScalingMetrics(
            timestamp=datetime.now(),
            cpu_usage=45.0,
            memory_usage=55.0,
            request_rate=10.0,
            response_time_ms=80.0,
            queue_depth=2,
            error_rate=0.5
        )
        
        scaler.record_metrics(metrics)
        assert len(scaler.metrics_history) > 0
    
    def test_scaling_status(self):
        """Test scaling status reporting."""
        scaler = get_advanced_auto_scaler()
        
        status = scaler.get_scaling_status()
        assert isinstance(status, dict)
        assert 'is_running' in status
        assert 'strategy' in status

class TestQuantumResearchFramework:
    """Test quantum research framework."""
    
    def test_quantum_processor(self):
        """Test quantum-inspired processor."""
        processor = QuantumInspiredProcessor(num_qubits=4)
        
        # Test circuit creation
        text_features = [0.5, -0.3, 0.8, 0.1]
        circuit = processor.create_sentiment_circuit(text_features)
        
        assert circuit.qubits == 4
        assert len(circuit.gates) > 0
        assert len(circuit.measurements) == 4
    
    def test_circuit_execution(self):
        """Test quantum circuit execution."""
        processor = QuantumInspiredProcessor(num_qubits=2)
        
        text_features = [0.6, -0.4]
        circuit = processor.create_sentiment_circuit(text_features)
        
        result = processor.execute_circuit(circuit, shots=100)
        
        assert 'measurements' in result
        assert 'quantum_metrics' in result
        assert result['shots'] == 100
    
    def test_research_manager(self):
        """Test research experiment manager."""
        manager = get_research_manager()
        
        # Test hypothesis registration
        hypothesis = ResearchHypothesis(
            id="test_hypothesis",
            title="Test Quantum Enhancement",
            description="Test hypothesis for quantum sentiment analysis",
            success_criteria={"accuracy": 0.05},
            baseline_metrics={"accuracy": 0.80},
            expected_improvement={"accuracy": 0.85}
        )
        
        manager.register_hypothesis(hypothesis)
        assert hypothesis.id in manager.hypotheses
    
    def test_experiment_setup(self):
        """Test setting up quantum sentiment experiment."""
        hypothesis_id = setup_quantum_sentiment_experiment()
        assert isinstance(hypothesis_id, str)
        
        manager = get_research_manager()
        assert hypothesis_id in manager.hypotheses

@pytest.mark.integration
class TestIntegrationScenarios:
    """Integration tests for complex scenarios."""
    
    def test_multilingual_compliance_workflow(self):
        """Test complete multilingual compliance workflow."""
        # Set Spanish language
        set_language('es')
        
        # Record consent
        compliance_mgr = get_compliance_manager()
        consent = compliance_mgr.record_consent(
            user_id="es_user_123",
            purpose=DataProcessingPurpose.SENTIMENT_ANALYSIS,
            granted=True
        )
        
        # Process data with compliance
        processing_id = compliance_mgr.process_data(
            user_id="es_user_123",
            data_type="texto",
            purpose=DataProcessingPurpose.SENTIMENT_ANALYSIS
        )
        
        assert processing_id is not None
        
        # Get translated message
        message = t('processing')
        assert message != 'processing'  # Should be Spanish
    
    def test_global_security_and_scaling(self):
        """Test global security with auto-scaling."""
        # Setup threat detection
        detector = get_threat_detector()
        
        # Simulate multiple requests from different regions
        for i in range(10):
            threat_level, threats, should_block = detector.analyze_request(
                ip=f"192.168.1.{i}",
                user_agent="Mozilla/5.0",
                path="/predict",
                payload={"text": f"test message {i}"}
            )
        
        # Record scaling metrics
        scaler = get_advanced_auto_scaler()
        metrics = ScalingMetrics(
            timestamp=datetime.now(),
            cpu_usage=85.0,  # High CPU
            memory_usage=70.0,
            request_rate=15.0,
            response_time_ms=120.0,
            queue_depth=8,
            error_rate=2.0
        )
        
        scaler.record_metrics(metrics)
        
        # Get security stats
        security_stats = detector.get_security_statistics()
        assert isinstance(security_stats, dict)
        
        # Get scaling status
        scaling_status = scaler.get_scaling_status()
        assert isinstance(scaling_status, dict)
    
    def test_research_with_caching_and_monitoring(self):
        """Test research framework with caching and health monitoring."""
        # Setup caching for research results
        cache_manager = get_cache_manager()
        
        @cache_result(ttl=300, tags=["research", "quantum"])
        def expensive_research_computation(experiment_id):
            time.sleep(0.1)  # Simulate computation
            return {"accuracy": 0.88, "quantum_advantage": 0.03}
        
        # Run computation
        result1 = expensive_research_computation("exp_001")
        result2 = expensive_research_computation("exp_001")  # Should be cached
        
        assert result1 == result2
        
        # Setup health monitoring for research system
        def research_health_check():
            return {"status": "healthy", "experiments_running": 1}
        
        research_checker = HealthChecker(
            name="research_system",
            component_type=ComponentType.MODEL_SERVICE,
            check_function=research_health_check,
            interval=5
        )
        
        monitor = get_health_monitor()
        monitor.add_health_checker(research_checker)
        
        # Get system health
        health_status = monitor.get_system_health()
        assert isinstance(health_status, dict)

@pytest.mark.performance
class TestPerformance:
    """Performance tests for advanced features."""
    
    def test_cache_performance(self):
        """Test cache performance under load."""
        cache = MemoryCache(max_size=1000, policy=CachePolicy.ADAPTIVE)
        
        start_time = time.time()
        
        # Perform many cache operations
        for i in range(1000):
            cache.set(f"key_{i}", f"value_{i}")
        
        for i in range(1000):
            cache.get(f"key_{i}")
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete within reasonable time
        assert duration < 1.0  # 1 second
        
        metrics = cache.get_metrics()
        assert metrics.hit_rate > 90.0  # Should have high hit rate
    
    def test_quantum_processing_performance(self):
        """Test quantum processing performance."""
        processor = QuantumInspiredProcessor(num_qubits=6)
        
        start_time = time.time()
        
        # Process multiple circuits
        for i in range(10):
            features = [0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i]
            circuit = processor.create_sentiment_circuit(features)
            result = processor.execute_circuit(circuit, shots=100)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should process circuits efficiently
        assert duration < 5.0  # 5 seconds for 10 circuits
        assert len(processor.quantum_circuits) == 10
    
    def test_concurrent_operations(self):
        """Test concurrent operations across systems."""
        import concurrent.futures
        
        def mixed_operations(thread_id):
            # Cache operations
            cache_manager = get_cache_manager()
            cache_manager.set(f"thread_{thread_id}", f"data_{thread_id}")
            
            # Security analysis
            detector = get_threat_detector()
            detector.analyze_request(
                ip=f"192.168.{thread_id}.1",
                user_agent="test",
                path="/test",
                payload={"data": f"thread_{thread_id}"}
            )
            
            # Scaling metrics
            scaler = get_advanced_auto_scaler()
            metrics = ScalingMetrics(
                timestamp=datetime.now(),
                cpu_usage=50.0 + thread_id,
                memory_usage=40.0,
                request_rate=5.0,
                response_time_ms=100.0,
                queue_depth=1,
                error_rate=0.1
            )
            scaler.record_metrics(metrics)
            
            return f"completed_{thread_id}"
        
        # Run concurrent operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(mixed_operations, i)
                for i in range(10)
            ]
            
            results = [future.result() for future in futures]
        
        assert len(results) == 10
        assert all("completed_" in result for result in results)

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])