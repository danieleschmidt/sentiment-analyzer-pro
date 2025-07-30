# Performance Optimization Guide

This guide provides comprehensive strategies for optimizing the sentiment analysis pipeline performance across different components and deployment scenarios.

## Performance Targets

| Component | Target Latency | Target Throughput | Memory Limit |
|-----------|---------------|-------------------|--------------|
| Model Loading | < 2s | N/A | < 500MB |
| Single Prediction | < 100ms | 10 req/s | < 50MB |
| Batch Processing | < 50ms/item | 1000 items/min | < 1GB |
| Web API | < 200ms | 100 req/s | < 256MB |

## Model Optimization

### 1. Model Selection Strategy
```python
# Performance vs Accuracy trade-offs
MODELS_BY_PERFORMANCE = {
    'naive_bayes': {
        'load_time': '< 100ms',
        'prediction': '< 10ms', 
        'memory': '< 50MB',
        'accuracy': '~85%'
    },
    'logistic_regression': {
        'load_time': '< 200ms',
        'prediction': '< 20ms',
        'memory': '< 100MB', 
        'accuracy': '~88%'
    },
    'lstm': {
        'load_time': '< 2s',
        'prediction': '< 50ms',
        'memory': '< 300MB',
        'accuracy': '~91%'
    },
    'transformer': {
        'load_time': '< 5s',
        'prediction': '< 100ms',
        'memory': '< 500MB',
        'accuracy': '~94%'
    }
}
```

### 2. Model Quantization
```python
# Reduce model size and inference time
from transformers import AutoModel
import torch

# 8-bit quantization
model = AutoModel.from_pretrained(
    "distilbert-base-uncased",
    torch_dtype=torch.int8,
    device_map="auto"
)

# Dynamic quantization
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

### 3. ONNX Conversion
```python
# Convert to ONNX for optimized inference
import torch.onnx
import onnxruntime

# Export model
torch.onnx.export(
    model,
    dummy_input,
    "sentiment_model.onnx",
    export_params=True,
    opset_version=11,
    input_names=['input'],
    output_names=['output']
)

# Load with ONNX Runtime
session = onnxruntime.InferenceSession("sentiment_model.onnx")
```

## Text Processing Optimization

### 1. Vectorized Operations
```python
import numpy as np
import pandas as pd

# Use vectorized pandas operations
def preprocess_batch(texts):
    df = pd.DataFrame({'text': texts})
    
    # Vectorized string operations
    df['clean_text'] = (df['text']
                       .str.lower()
                       .str.replace(r'[^\w\s]', '', regex=True)
                       .str.replace(r'\s+', ' ', regex=True)
                       .str.strip())
    
    return df['clean_text'].tolist()
```

### 2. Efficient Tokenization
```python
from transformers import AutoTokenizer

# Initialize tokenizer once
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Batch tokenization with padding
def tokenize_batch(texts, max_length=128):
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
```

### 3. Memory-Mapped File Reading
```python
import numpy as np
import pandas as pd

# Use memory mapping for large datasets
def load_large_dataset(filepath):
    # Memory-mapped reading
    with open(filepath, 'r') as f:
        # Process in chunks to avoid memory issues
        chunk_size = 10000
        for chunk in pd.read_csv(f, chunksize=chunk_size):
            yield chunk
```

## Web API Optimization

### 1. Response Caching
```python
from functools import lru_cache
import hashlib

class PredictionCache:
    def __init__(self, max_size=1000, ttl=300):
        self.cache = {}
        self.max_size = max_size
        self.ttl = ttl
    
    def get_cache_key(self, text):
        return hashlib.md5(text.encode()).hexdigest()
    
    @lru_cache(maxsize=1000)
    def predict_cached(self, text_hash, text):
        # Actual prediction logic here
        return self.model.predict([text])[0]
```

### 2. Async Request Handling
```python
from flask import Flask
import asyncio
import concurrent.futures

app = Flask(__name__)
executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

@app.route('/predict', methods=['POST'])
async def predict_async():
    text = request.json['text']
    
    # Run prediction in thread pool
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        executor, 
        predict_function, 
        text
    )
    
    return {'prediction': result}
```

### 3. Connection Pooling
```python
from flask import Flask
from werkzeug.serving import WSGIRequestHandler

# Configure connection pooling
class OptimizedRequestHandler(WSGIRequestHandler):
    def setup(self):
        super().setup()
        # Keep connections alive
        self.connection.setsockopt(
            socket.SOL_SOCKET, 
            socket.SO_KEEPALIVE, 1
        )
```

## Memory Optimization

### 1. Memory Profiling
```python
import tracemalloc
from memory_profiler import profile

# Memory tracking
tracemalloc.start()

@profile
def memory_intensive_function():
    # Your code here
    pass

# Get memory usage
current, peak = tracemalloc.get_traced_memory()
print(f"Current: {current / 1024 / 1024:.1f}MB")
print(f"Peak: {peak / 1024 / 1024:.1f}MB")
```

### 2. Garbage Collection Optimization
```python
import gc

# Optimize garbage collection
gc.set_threshold(700, 10, 10)  # Tune thresholds
gc.collect()  # Force collection at strategic points

# Disable GC during critical sections
gc.disable()
# ... critical code ...
gc.enable()
```

### 3. Memory Pool Management
```python
import numpy as np

# Pre-allocate memory pools
class MemoryPool:
    def __init__(self, max_size=1000):
        self.pool = np.empty((max_size, 128), dtype=np.float32)
        self.used = 0
    
    def get_array(self, size):
        if self.used + size <= self.pool.shape[0]:
            result = self.pool[self.used:self.used + size]
            self.used += size
            return result
        return np.empty((size, 128), dtype=np.float32)
```

## Batch Processing Optimization

### 1. Parallel Processing
```python
from multiprocessing import Pool
import concurrent.futures

def process_batch_parallel(texts, batch_size=100):
    # Split into batches
    batches = [texts[i:i+batch_size] 
              for i in range(0, len(texts), batch_size)]
    
    # Process in parallel
    with Pool() as pool:
        results = pool.map(predict_batch, batches)
    
    # Flatten results
    return [item for batch in results for item in batch]
```

### 2. GPU Acceleration
```python
import torch

# Use GPU when available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def predict_gpu_batch(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    return outputs.logits.cpu().numpy()
```

## Monitoring and Profiling

### 1. Performance Metrics Collection
```python
import time
import psutil
from prometheus_client import Counter, Histogram

# Metrics
prediction_counter = Counter('predictions_total', 'Total predictions')
prediction_duration = Histogram('prediction_duration_seconds', 'Prediction latency')

def track_performance(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        memory_before = psutil.Process().memory_info().rss
        
        result = func(*args, **kwargs)
        
        duration = time.time() - start_time
        memory_after = psutil.Process().memory_info().rss
        
        prediction_counter.inc()
        prediction_duration.observe(duration)
        
        return result
    return wrapper
```

### 2. Continuous Benchmarking
```bash
# Add to CI/CD pipeline
pytest tests/benchmarks/ --benchmark-only --benchmark-json=benchmark.json

# Performance regression detection
python scripts/check_performance_regression.py benchmark.json
```

## Deployment Optimization

### 1. Container Optimization
```dockerfile
# Multi-stage build for smaller images
FROM python:3.9-slim as builder
COPY requirements.txt .
RUN pip install --user -r requirements.txt

FROM python:3.9-slim
COPY --from=builder /root/.local /root/.local
COPY . .

# Use gunicorn for production
CMD ["gunicorn", "--workers", "4", "--bind", "0.0.0.0:5000", "src.webapp:app"]
```

### 2. Load Balancing Configuration
```yaml
# docker-compose.yml
version: '3.8'
services:
  sentiment-api:
    build: .
    deploy:
      replicas: 4
      resources:
        limits:
          memory: 512M
        reservations:
          memory: 256M
  
  nginx:
    image: nginx
    ports:
      - "80:80"
    depends_on:
      - sentiment-api
```

## Performance Testing

### 1. Load Testing
```python
import requests
import concurrent.futures
import time

def load_test(url, num_requests=100, concurrency=10):
    def make_request():
        response = requests.post(
            url, 
            json={'text': 'Test sentiment analysis'}
        )
        return response.status_code, response.elapsed.total_seconds()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(make_request) for _ in range(num_requests)]
        results = [future.result() for future in futures]
    
    # Analyze results
    latencies = [result[1] for result in results]
    print(f"Average latency: {sum(latencies) / len(latencies):.3f}s")
    print(f"95th percentile: {sorted(latencies)[int(0.95 * len(latencies))]:.3f}s")
```

### 2. Memory Stress Testing
```python
def memory_stress_test(batch_sizes=[100, 500, 1000, 5000]):
    for batch_size in batch_sizes:
        print(f"Testing batch size: {batch_size}")
        
        # Generate test data
        texts = ["Test text for analysis"] * batch_size
        
        # Measure memory before
        memory_before = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Run prediction
        start_time = time.time()
        results = predict_batch(texts)
        duration = time.time() - start_time
        
        # Measure memory after
        memory_after = psutil.Process().memory_info().rss / 1024 / 1024
        
        print(f"Duration: {duration:.2f}s")
        print(f"Memory usage: {memory_after - memory_before:.1f}MB")
        print(f"Throughput: {batch_size / duration:.1f} items/sec")
        print("---")
```

## Best Practices

1. **Profile Before Optimizing**: Always measure performance before making changes
2. **Optimize Bottlenecks**: Focus on the slowest components first
3. **Memory vs Speed Trade-offs**: Balance memory usage with execution speed
4. **Cache Strategically**: Cache expensive computations with appropriate TTL
5. **Monitor Continuously**: Set up alerts for performance regressions
6. **Test Under Load**: Ensure optimizations work under realistic conditions

For automated performance monitoring, see [PERFORMANCE_MONITORING.md](PERFORMANCE_MONITORING.md).