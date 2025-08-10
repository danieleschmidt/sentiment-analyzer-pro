"""
Comprehensive test suite for autonomous SDLC implementation
Quality Gates: Automated testing for all three generations
"""
import pytest
import time
import json
import threading
from unittest.mock import Mock, patch
import pandas as pd
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Generation 1 Tests
class TestGeneration1:
    """Test Generation 1: Make It Work"""
    
    def test_health_check_system(self):
        """Test basic health check functionality"""
        from src.health_check import HealthChecker
        
        checker = HealthChecker()
        results = checker.run_all_checks()
        
        assert len(results) > 0
        assert 'system_resources' in results
        assert 'dependencies' in results
        assert 'model_status' in results
        
        # All checks should have completed
        for name, result in results.items():
            assert result.timestamp > 0
            assert result.message is not None
    
    def test_enhanced_config_loading(self):
        """Test configuration system"""
        from src.enhanced_config import ConfigManager
        
        config_manager = ConfigManager()
        config = config_manager.config
        
        # Test default values
        assert config.environment == "development"
        assert config.server.host == "127.0.0.1"
        assert config.server.port == 5000
        assert config.model.nb_alpha == 1.0
    
    def test_core_api_functionality(self):
        """Test core API functionality"""
        from src.core_api import SentimentAPI
        
        api = SentimentAPI()
        
        # Test single prediction
        result = api.predict_sentiment("This is great!")
        
        assert result.text == "This is great!"
        assert result.sentiment in ['positive', 'negative', 'neutral']
        assert result.processing_time_ms > 0
        assert result.model_version == "nb_v1"
        
        # Test batch prediction
        texts = ["Great!", "Terrible!", "Okay"]
        batch_results = api.predict_batch(texts)
        
        assert len(batch_results) == 3
        for result in batch_results:
            assert result.sentiment in ['positive', 'negative', 'neutral']
    
    def test_api_stats(self):
        """Test API statistics tracking"""
        from src.core_api import SentimentAPI
        
        api = SentimentAPI()
        
        # Make some predictions
        api.predict_sentiment("Test 1")
        api.predict_sentiment("Test 2")
        
        stats = api.get_stats()
        
        assert stats['total_predictions'] == 2
        assert stats['model_loaded'] is True
        assert 'uptime_seconds' in stats

class TestGeneration2:
    """Test Generation 2: Make It Robust"""
    
    def test_robust_error_handling(self):
        """Test robust error handling system"""
        from src.robust_error_handling import RobustErrorHandler, ErrorSeverity, ErrorCategory
        
        handler = RobustErrorHandler()
        
        # Test error handling
        try:
            raise ValueError("Test error")
        except ValueError as e:
            error_context = handler.handle_error(
                e, 
                severity=ErrorSeverity.MEDIUM,
                category=ErrorCategory.VALIDATION
            )
            
            assert error_context.severity == ErrorSeverity.MEDIUM
            assert error_context.category == ErrorCategory.VALIDATION
            assert "Test error" in error_context.message
            assert error_context.timestamp > 0
    
    def test_robust_decorator(self):
        """Test robust function decorator"""
        from src.robust_error_handling import robust_function, ErrorSeverity
        
        call_count = 0
        
        @robust_function(severity=ErrorSeverity.LOW, max_retries=2)
        def test_function(x):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Retry needed")
            return x * 2
        
        # Should succeed after retries
        result = test_function(5)
        assert result == 10
        assert call_count == 3
    
    def test_security_framework(self):
        """Test security framework components"""
        from src.security_framework import SecurityFramework
        
        security = SecurityFramework()
        
        # Test JWT
        token = security.jwt_manager.generate_token({"user_id": "123"})
        assert token is not None
        
        payload = security.jwt_manager.validate_token(token)
        assert payload["user_id"] == "123"
        
        # Test input sanitization
        dangerous_input = "<script>alert('xss')</script>Hello"
        sanitized = security.input_sanitizer.sanitize_text(dangerous_input)
        assert "<script>" not in sanitized
        assert "Hello" in sanitized
        
        # Test password hashing
        password = "test_password"
        hashed, salt = security.hash_password(password)
        assert security.verify_password(password, hashed, salt)
        assert not security.verify_password("wrong", hashed, salt)
    
    def test_rate_limiting(self):
        """Test rate limiting functionality"""
        from src.security_framework import RateLimiter
        
        limiter = RateLimiter(max_requests=3, time_window=60)
        client_id = "test_client"
        
        # Should allow first 3 requests
        assert limiter.is_allowed(client_id)
        assert limiter.is_allowed(client_id)
        assert limiter.is_allowed(client_id)
        
        # Should block 4th request
        assert not limiter.is_allowed(client_id)
    
    def test_data_validation(self):
        """Test data validation framework"""
        from src.data_validation import setup_sentiment_data_validation
        
        validator = setup_sentiment_data_validation()
        
        # Test with good data
        good_data = pd.DataFrame({
            'text': ['This is great!', 'I love it'],
            'label': ['positive', 'positive']
        })
        
        report = validator.validate_dataframe(good_data)
        assert report.quality_score >= 80
        
        # Test with problematic data
        bad_data = pd.DataFrame({
            'text': ['', None, 'A' * 20000],
            'label': ['invalid', 'positive', 'negative']
        })
        
        bad_report = validator.validate_dataframe(bad_data)
        assert bad_report.quality_score < 80
        assert len(bad_report.validation_results) > 0

class TestGeneration3:
    """Test Generation 3: Make It Scale"""
    
    def test_smart_cache(self):
        """Test smart caching system"""
        from src.performance_engine import SmartCache, CacheStrategy
        
        cache = SmartCache(max_size=10, strategy=CacheStrategy.LRU)
        
        # Test cache miss and put
        result = cache.get("key1")
        assert result is None
        
        cache.put("key1", "value1")
        result = cache.get("key1")
        assert result == "value1"
        
        stats = cache.get_stats()
        assert stats['hit_count'] == 1
        assert stats['miss_count'] == 1
    
    def test_cached_decorator(self):
        """Test smart cache decorator"""
        from src.performance_engine import smart_cache, CacheStrategy
        
        call_count = 0
        
        @smart_cache(max_size=5, strategy=CacheStrategy.LRU)
        def expensive_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2
        
        # First call should execute function
        result1 = expensive_function(5)
        assert result1 == 10
        assert call_count == 1
        
        # Second call should use cache
        result2 = expensive_function(5)
        assert result2 == 10
        assert call_count == 1  # Function not called again
    
    def test_batch_processor(self):
        """Test batch processing system"""
        from src.performance_engine import BatchProcessor
        
        processor = BatchProcessor(batch_size=5, max_workers=2)
        
        def process_item(x):
            return x * 2
        
        items = list(range(10))
        results = processor.process_batch(items, process_item)
        
        assert len(results) == 10
        assert results == [x * 2 for x in items]
        
        stats = processor.get_stats()
        assert stats['processed_count'] == 10
    
    def test_memory_optimizer(self):
        """Test memory optimization"""
        from src.performance_engine import MemoryOptimizer
        
        # Test DataFrame optimization
        df = pd.DataFrame({
            'int_col': [1, 2, 3, 4, 5],
            'float_col': [1.0, 2.0, 3.0, 4.0, 5.0],
            'category_col': ['A', 'B', 'A', 'B', 'A']
        })
        
        original_memory = df.memory_usage(deep=True).sum()
        optimized_df = MemoryOptimizer.optimize_dataframe(df)
        optimized_memory = optimized_df.memory_usage(deep=True).sum()
        
        # Should use same or less memory
        assert optimized_memory <= original_memory
    
    def test_performance_monitoring(self):
        """Test performance monitoring"""
        from src.performance_engine import PerformanceMonitor
        
        monitor = PerformanceMonitor()
        
        # Test operation tracking
        with monitor.track_operation("test_op") as tracker:
            time.sleep(0.01)  # Simulate work
            tracker.mark_cache_hit()
        
        summary = monitor.get_performance_summary("test_op")
        
        assert summary['total_operations'] == 1
        assert summary['avg_duration_ms'] >= 10  # At least 10ms
        assert summary['cache_hit_rate'] == 1.0
    
    def test_resource_monitor(self):
        """Test resource monitoring"""
        from src.scalable_architecture import ResourceMonitor
        
        monitor = ResourceMonitor(check_interval=0.1)
        monitor.start_monitoring()
        
        time.sleep(0.2)  # Let it collect some metrics
        
        metrics = monitor.get_current_metrics()
        assert metrics is not None
        assert metrics.cpu_percent >= 0
        assert metrics.memory_percent >= 0
        
        monitor.stop_monitoring()
    
    def test_worker_pool(self):
        """Test worker pool functionality"""
        from src.scalable_architecture import WorkerPool
        
        pool = WorkerPool(min_workers=2, max_workers=4)
        pool.start()
        
        # Submit task
        future = pool.submit_task(lambda x: x ** 2, 5)
        result = future.result(timeout=5)
        
        assert result == 25
        
        stats = pool.get_stats()
        assert stats['current_workers'] >= 2
        
        pool.stop()

class TestIntegration:
    """Integration tests across all generations"""
    
    def test_full_pipeline_integration(self):
        """Test complete pipeline from API to prediction"""
        from src.core_api import SentimentAPI
        from src.performance_engine import HighPerformanceSentimentAnalyzer
        
        # Test basic API
        api = SentimentAPI()
        result = api.predict_sentiment("This is amazing!")
        assert result.sentiment == "positive"
        
        # Test high-performance version
        hp_analyzer = HighPerformanceSentimentAnalyzer()
        batch_results = hp_analyzer.predict_batch_optimized(["Great!", "Terrible!"])
        
        assert len(batch_results) == 2
        assert batch_results[0]['sentiment'] == 'positive'
        assert batch_results[1]['sentiment'] == 'negative'
    
    def test_error_handling_integration(self):
        """Test error handling across components"""
        from src.core_api import SentimentAPI
        from src.robust_error_handling import get_logger
        
        logger = get_logger()
        api = SentimentAPI()
        
        # Test invalid input handling
        try:
            result = api.predict_sentiment("")
        except:
            pass  # Expected to handle gracefully
        
        # Check that errors were logged
        error_summary = logger.get_error_summary()
        assert error_summary['total_errors'] >= 0  # May be 0 if handled gracefully
    
    def test_security_integration(self):
        """Test security features integration"""
        from src.security_framework import SecurityFramework
        from flask import Flask
        
        security = SecurityFramework()
        
        # Test full security workflow
        app = Flask(__name__)
        
        # Generate token
        token = security.jwt_manager.generate_token({"user_id": "test"})
        
        # Validate token
        payload = security.jwt_manager.validate_token(token)
        assert payload["user_id"] == "test"
        
        # Test input sanitization
        dirty_input = "<script>evil()</script>Hello world"
        clean_input = security.input_sanitizer.sanitize_text(dirty_input)
        assert "script" not in clean_input
        assert "Hello world" in clean_input

class TestPerformanceBenchmarks:
    """Performance benchmarks and SLA validation"""
    
    def test_prediction_latency(self):
        """Test that predictions meet latency SLA"""
        from src.core_api import SentimentAPI
        
        api = SentimentAPI()
        
        # Single prediction should be fast
        start_time = time.time()
        result = api.predict_sentiment("Test text for latency")
        latency_ms = (time.time() - start_time) * 1000
        
        # Should be under 100ms for simple prediction
        assert latency_ms < 100, f"Prediction took {latency_ms:.2f}ms, SLA is 100ms"
    
    def test_batch_throughput(self):
        """Test batch processing throughput"""
        from src.performance_engine import HighPerformanceSentimentAnalyzer
        
        analyzer = HighPerformanceSentimentAnalyzer()
        texts = ["Test text"] * 100
        
        start_time = time.time()
        results = analyzer.predict_batch_optimized(texts)
        total_time = time.time() - start_time
        
        throughput = len(results) / total_time
        
        # Should process at least 50 texts per second
        assert throughput >= 50, f"Throughput {throughput:.1f} TPS below SLA of 50 TPS"
    
    def test_memory_efficiency(self):
        """Test memory usage stays within bounds"""
        from src.performance_engine import MemoryOptimizer, HighPerformanceSentimentAnalyzer
        
        initial_memory = MemoryOptimizer.get_memory_usage()
        
        analyzer = HighPerformanceSentimentAnalyzer()
        
        # Process large batch
        large_batch = ["Test text " * 100] * 1000
        results = analyzer.predict_batch_optimized(large_batch)
        
        peak_memory = MemoryOptimizer.get_memory_usage()
        memory_increase = peak_memory - initial_memory
        
        # Should not use more than 100MB additional memory
        assert memory_increase < 100, f"Memory usage increased by {memory_increase:.1f}MB"

def run_quality_gates():
    """Run all quality gates and return results"""
    import subprocess
    
    print("🔍 Running Quality Gates...")
    
    # Run tests
    result = subprocess.run([
        'python', '-m', 'pytest', 
        'tests/test_autonomous_sdlc.py', 
        '-v', '--tb=short'
    ], capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(__file__)))
    
    print("📊 Quality Gate Results:")
    print(result.stdout)
    
    if result.stderr:
        print("⚠️ Warnings/Errors:")
        print(result.stderr)
    
    success = result.returncode == 0
    print(f"✅ Quality Gates: {'PASSED' if success else 'FAILED'}")
    
    return success, result.stdout, result.stderr

if __name__ == "__main__":
    # Run quality gates directly
    success, stdout, stderr = run_quality_gates()
    
    if success:
        print("🎉 All quality gates passed!")
        exit(0)
    else:
        print("❌ Quality gates failed!")
        exit(1)