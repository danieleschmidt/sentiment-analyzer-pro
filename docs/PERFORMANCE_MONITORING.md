# Performance Monitoring and Profiling

This document outlines performance monitoring and profiling capabilities for the Sentiment Analyzer Pro application.

## Performance Testing Framework

### 1. Benchmarking Setup

The project includes comprehensive model comparison and benchmarking in `src/model_comparison.py`. For additional performance testing:

```python
# Performance test configuration
import time
import psutil
import memory_profiler
from functools import wraps

def performance_monitor(func):
    """Decorator to monitor function performance"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Memory and CPU monitoring
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        start_cpu = process.cpu_percent()
        start_time = time.time()
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        end_memory = process.memory_info().rss / 1024 / 1024  # MB
        end_cpu = process.cpu_percent()
        
        print(f"Function: {func.__name__}")
        print(f"Execution time: {end_time - start_time:.4f} seconds")
        print(f"Memory usage: {end_memory - start_memory:.2f} MB")
        print(f"CPU usage: {end_cpu - start_cpu:.2f}%")
        
        return result
    return wrapper
```

### 2. Load Testing for Web API

Create load testing with locust:

```python
# tests/performance/locustfile.py
from locust import HttpUser, task, between

class SentimentAnalysisUser(HttpUser):
    wait_time = between(1, 3)
    
    @task
    def predict_sentiment(self):
        payload = {"text": "This is a great product!"}
        self.client.post("/predict", json=payload)
    
    @task(3)
    def health_check(self):
        self.client.get("/")
```

### 3. Memory Profiling

Using memory_profiler for detailed memory analysis:

```python
# Performance profiling script
@memory_profiler.profile
def profile_model_training():
    from src.transformer_trainer import TransformerTrainer, TransformerConfig
    
    config = TransformerConfig(num_epochs=1, batch_size=8)
    trainer = TransformerTrainer(config)
    trainer.train("data/sample_reviews.csv")
```

## Monitoring Configuration

### 1. Application Metrics

Add to Flask webapp for production monitoring:

```python
# src/webapp.py - Add metrics collection
import time
from collections import defaultdict

class MetricsCollector:
    def __init__(self):
        self.request_count = 0
        self.response_times = []
        self.error_count = 0
        self.prediction_count = 0
    
    def record_request(self, duration, endpoint, status_code):
        self.request_count += 1
        self.response_times.append(duration)
        if status_code >= 400:
            self.error_count += 1
        if endpoint == '/predict':
            self.prediction_count += 1
```

### 2. Database Performance (if applicable)

```python
# Query performance monitoring
import logging
import time

class QueryPerformanceMonitor:
    def __init__(self):
        self.logger = logging.getLogger('query_performance')
    
    def log_query(self, query, duration):
        if duration > 0.1:  # Log slow queries > 100ms
            self.logger.warning(f"Slow query: {query} took {duration:.3f}s")
```

## Performance Testing Suite

### 1. Automated Performance Tests

```python
# tests/performance/test_performance.py
import pytest
import time
from src.predict import predict_sentiment

class TestPerformance:
    
    def test_prediction_speed(self):
        """Test single prediction performance"""
        text = "This is a test review"
        
        start_time = time.time()
        result = predict_sentiment(text)
        duration = time.time() - start_time
        
        assert duration < 1.0  # Should complete in under 1 second
        assert result in ['positive', 'negative']
    
    def test_batch_prediction_speed(self):
        """Test batch prediction performance"""
        texts = ["Great product!" for _ in range(100)]
        
        start_time = time.time()
        results = [predict_sentiment(text) for text in texts]
        duration = time.time() - start_time
        
        assert duration < 10.0  # Should complete 100 predictions in under 10 seconds
        assert len(results) == 100
    
    @pytest.mark.slow
    def test_memory_usage(self):
        """Test memory usage doesn't exceed limits"""
        import psutil
        process = psutil.Process()
        
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Perform memory-intensive operation
        from src.model_comparison import benchmark_models
        benchmark_models("data/sample_reviews.csv")
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        assert memory_increase < 500  # Should not use more than 500MB additional
```

### 2. CI Integration

Add to Makefile:

```makefile
# Performance testing targets
performance-test: ## Run performance tests
	@echo "$(GREEN)Running performance tests...$(NC)"
	@$(PYTEST) tests/performance/ -v --tb=short

benchmark: ## Run benchmarks and generate report
	@echo "$(GREEN)Running benchmarks...$(NC)"
	@$(PYTHON) -m src.model_comparison --benchmark --output benchmark_results.json

load-test: ## Run load tests (requires locust)
	@echo "$(GREEN)Starting load test...$(NC)"
	@locust -f tests/performance/locustfile.py --headless -u 10 -r 2 -t 30s --host http://localhost:5000
```

## Profiling Tools Integration

### 1. cProfile Integration

```python
# scripts/profile_app.py
import cProfile
import pstats
from src.model_comparison import benchmark_models

def profile_benchmarks():
    profiler = cProfile.Profile()
    profiler.enable()
    
    benchmark_models("data/sample_reviews.csv")
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats()
    stats.dump_stats('benchmark_profile.prof')

if __name__ == "__main__":
    profile_benchmarks()
```

### 2. py-spy Integration

For production profiling without code changes:

```bash
# Profile running application
py-spy record -o profile.svg --pid $(pgrep -f "python.*webapp")

# Profile for 30 seconds
py-spy record -o profile.svg --duration 30 --pid $(pgrep -f "python.*webapp")
```

## Monitoring Dashboards

### 1. Simple HTTP Monitoring

```python
# monitoring/simple_monitor.py
import requests
import time
import json
from datetime import datetime

def monitor_health():
    endpoints = [
        'http://localhost:5000/',
        'http://localhost:5000/version',
        'http://localhost:5000/metrics'
    ]
    
    results = {}
    for endpoint in endpoints:
        try:
            start = time.time()
            response = requests.get(endpoint, timeout=5)
            duration = time.time() - start
            
            results[endpoint] = {
                'status_code': response.status_code,
                'response_time': duration,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            results[endpoint] = {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    return results
```

### 2. Performance Metrics Collection

The webapp already includes basic metrics at `/metrics`. Enhance with:

```python
# Extended metrics for monitoring
{
    "requests_total": metrics.request_count,
    "predictions_total": metrics.prediction_count,
    "errors_total": metrics.error_count,
    "avg_response_time": sum(metrics.response_times) / len(metrics.response_times),
    "max_response_time": max(metrics.response_times),
    "uptime_seconds": time.time() - app_start_time,
    "memory_usage_mb": psutil.Process().memory_info().rss / 1024 / 1024,
    "cpu_percent": psutil.Process().cpu_percent()
}
```

## Implementation Requirements

### Dependencies for Performance Monitoring

Add to pyproject.toml:

```toml
[project.optional-dependencies]
performance = [
    "memory-profiler",
    "psutil",
    "locust",
    "py-spy",
]
```

### Installation

```bash
# Install performance monitoring tools
pip install -e .[performance]

# Or install individually
pip install memory-profiler psutil locust py-spy
```

## Usage Examples

### Running Performance Tests

```bash
# Quick performance check
make performance-test

# Full benchmark suite
make benchmark

# Load testing (requires running server)
python -m src.webapp &
make load-test

# Memory profiling
python -m memory_profiler scripts/profile_memory.py

# CPU profiling
python -m cProfile -o profile.prof -m src.model_comparison
```

### Monitoring in Production

```python
# Enable performance monitoring in production
from monitoring.simple_monitor import monitor_health

# Run periodic health checks
results = monitor_health()
print(json.dumps(results, indent=2))
```

This performance monitoring framework provides comprehensive insights into application performance across different dimensions: CPU usage, memory consumption, response times, and throughput.