# Performance Profiling and Optimization

This directory contains performance analysis tools and configurations for optimizing the sentiment analysis pipeline.

## Quick Start

1. **Install profiling dependencies**:
   ```bash
   pip install py-spy memory-profiler line-profiler pytest-benchmark
   ```

2. **Run performance benchmarks**:
   ```bash
   make benchmark
   ```

3. **Generate CPU profile**:
   ```bash
   make profile-cpu
   ```

4. **Check memory usage**:
   ```bash
   make profile-memory
   ```

## Profiling Tools

### CPU Profiling
- **py-spy**: Low-overhead sampling profiler
- **cProfile**: Built-in deterministic profiler
- **line_profiler**: Line-by-line profiling

### Memory Profiling
- **memory_profiler**: Monitor memory usage over time
- **tracemalloc**: Built-in memory tracker
- **pympler**: Advanced memory analysis

### Benchmarking
- **pytest-benchmark**: Automated performance testing
- **timeit**: Micro-benchmarking utilities

## Performance Targets

| Component | Target | Current | Status |
|-----------|--------|---------|--------|
| Model Loading | < 2s | TBD | 🔄 |
| Single Prediction | < 100ms | TBD | 🔄 |
| Batch Processing (100) | < 5s | TBD | 🔄 |
| Memory Usage | < 512MB | TBD | 🔄 |

## Optimization Strategies

### Model Performance
- Model quantization for reduced memory footprint
- ONNX conversion for faster inference
- Batch prediction optimization
- Caching for repeated predictions

### Data Processing
- Vectorized text preprocessing
- Efficient tokenization pipelines
- Memory-mapped file reading for large datasets
- Parallel processing for batch operations

### Web API Performance
- Response caching with TTL
- Connection pooling
- Async request handling
- Rate limiting and throttling

## Monitoring Integration

Performance metrics are automatically collected and exported to:
- **Prometheus**: Real-time metrics collection
- **Grafana**: Dashboards and visualization
- **Application logs**: Structured performance data

See [../monitoring/README.md](../monitoring/README.md) for monitoring setup details.

## Continuous Performance Testing

Performance tests run automatically in CI/CD:
- Benchmark tests for critical paths
- Memory leak detection
- Performance regression alerts
- Load testing for web endpoints

## Troubleshooting

### Common Performance Issues
1. **Slow model loading**: Check model size and disk I/O
2. **High memory usage**: Profile memory allocation patterns  
3. **Slow predictions**: Analyze tokenization and inference bottlenecks
4. **API latency**: Check network, serialization, and processing time

### Debugging Commands
```bash
# Profile a specific script
python -m cProfile -o profile.stats script.py

# Monitor memory usage
python -m memory_profiler script.py

# Live CPU profiling
py-spy top --pid $(pgrep -f python)

# Memory snapshot
python -c "import tracemalloc; tracemalloc.start(); # your code here"
```

For detailed profiling guides, see the individual tool configurations in this directory.