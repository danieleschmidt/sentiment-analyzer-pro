"""
Photonic-MLIR Bridge - Comprehensive Performance Suite

This module provides advanced performance monitoring, benchmarking, profiling,
and optimization capabilities for the photonic-MLIR synthesis bridge.
"""

import time
import logging
import threading
import cProfile
import pstats
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Tuple
from pathlib import Path
import json
import statistics
import psutil
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
import functools

logger = logging.getLogger(__name__)


class BenchmarkType(Enum):
    """Types of benchmarks."""
    SYNTHESIS_THROUGHPUT = "synthesis_throughput"
    VALIDATION_PERFORMANCE = "validation_performance"
    OPTIMIZATION_EFFICIENCY = "optimization_efficiency"
    MEMORY_USAGE = "memory_usage"
    CONCURRENT_PROCESSING = "concurrent_processing"
    CACHE_PERFORMANCE = "cache_performance"
    ERROR_HANDLING_OVERHEAD = "error_handling_overhead"


class PerformanceMetric(Enum):
    """Performance metrics to track."""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    CACHE_HIT_RATE = "cache_hit_rate"
    ERROR_RATE = "error_rate"
    QUEUE_LENGTH = "queue_length"
    RESPONSE_TIME = "response_time"


@dataclass
class BenchmarkResult:
    """Result of a performance benchmark."""
    benchmark_type: BenchmarkType
    timestamp: float
    duration: float
    metrics: Dict[str, Any]
    statistics: Dict[str, float]
    metadata: Dict[str, Any]


@dataclass
class PerformanceProfile:
    """Performance profile data."""
    function_name: str
    total_time: float
    cumulative_time: float
    call_count: int
    time_per_call: float
    filename: str
    line_number: int


class PerformanceMonitor:
    """Advanced performance monitoring system."""
    
    def __init__(self):
        self.metrics_history: Dict[str, List[Tuple[float, Any]]] = {}
        self.monitoring_active = False
        self.monitoring_thread = None
        self.lock = threading.RLock()
        
        # Performance counters
        self.operation_counts = {}
        self.operation_times = {}
        self.error_counts = {}
        
        # Memory tracking
        self.memory_snapshots = []
        self.peak_memory_usage = 0
        
        # Cache statistics
        self.cache_hits = 0
        self.cache_misses = 0
        
        self._initialize_monitoring()
    
    def _initialize_monitoring(self):
        """Initialize performance monitoring."""
        # Start memory tracking
        tracemalloc.start()
        
        # Initialize metric collections
        for metric in PerformanceMetric:
            self.metrics_history[metric.value] = []
        
        logger.info("Performance monitoring initialized")
    
    def start_monitoring(self, interval: float = 1.0):
        """Start continuous performance monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True,
            name="PerformanceMonitor"
        )
        self.monitoring_thread.start()
        
        logger.info(f"Performance monitoring started with {interval}s interval")
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self, interval: float):
        """Main performance monitoring loop."""
        while self.monitoring_active:
            try:
                timestamp = time.time()
                
                # Collect system metrics
                cpu_usage = psutil.cpu_percent(interval=None)
                memory_info = psutil.virtual_memory()
                
                # Record metrics
                self.record_metric(PerformanceMetric.CPU_USAGE, cpu_usage, timestamp)
                self.record_metric(PerformanceMetric.MEMORY_USAGE, memory_info.percent, timestamp)
                
                # Track peak memory
                current_memory, peak = tracemalloc.get_traced_memory()
                if peak > self.peak_memory_usage:
                    self.peak_memory_usage = peak
                
                self.memory_snapshots.append((timestamp, current_memory, peak))
                
                # Calculate cache hit rate
                total_cache_ops = self.cache_hits + self.cache_misses
                if total_cache_ops > 0:
                    hit_rate = self.cache_hits / total_cache_ops
                    self.record_metric(PerformanceMetric.CACHE_HIT_RATE, hit_rate, timestamp)
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                time.sleep(interval * 2)
    
    def record_metric(self, metric: PerformanceMetric, value: Any, timestamp: float = None):
        """Record a performance metric."""
        if timestamp is None:
            timestamp = time.time()
        
        with self.lock:
            if metric.value not in self.metrics_history:
                self.metrics_history[metric.value] = []
            
            self.metrics_history[metric.value].append((timestamp, value))
            
            # Keep history manageable
            if len(self.metrics_history[metric.value]) > 10000:
                self.metrics_history[metric.value] = self.metrics_history[metric.value][-5000:]
    
    def record_operation(self, operation_name: str, duration: float, success: bool = True):
        """Record operation performance."""
        with self.lock:
            # Count operations
            if operation_name not in self.operation_counts:
                self.operation_counts[operation_name] = 0
                self.operation_times[operation_name] = []
                self.error_counts[operation_name] = 0
            
            self.operation_counts[operation_name] += 1
            self.operation_times[operation_name].append(duration)
            
            if not success:
                self.error_counts[operation_name] += 1
            
            # Record metrics
            self.record_metric(PerformanceMetric.LATENCY, duration)
            
            # Calculate throughput (operations per second)
            if len(self.operation_times[operation_name]) >= 10:
                recent_times = self.operation_times[operation_name][-10:]
                avg_time = statistics.mean(recent_times)
                throughput = 1.0 / avg_time if avg_time > 0 else 0
                self.record_metric(PerformanceMetric.THROUGHPUT, throughput)
    
    def record_cache_operation(self, hit: bool):
        """Record cache operation."""
        if hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance monitoring summary."""
        with self.lock:
            summary = {
                "timestamp": time.time(),
                "monitoring_active": self.monitoring_active,
                "peak_memory_mb": self.peak_memory_usage / (1024 * 1024),
                "operation_statistics": {},
                "metric_summaries": {},
                "cache_statistics": {
                    "hits": self.cache_hits,
                    "misses": self.cache_misses,
                    "hit_rate": self.cache_hits / max(self.cache_hits + self.cache_misses, 1)
                }
            }
            
            # Operation statistics
            for op_name in self.operation_counts:
                times = self.operation_times[op_name]
                if times:
                    summary["operation_statistics"][op_name] = {
                        "count": self.operation_counts[op_name],
                        "errors": self.error_counts[op_name],
                        "error_rate": self.error_counts[op_name] / self.operation_counts[op_name],
                        "avg_time": statistics.mean(times),
                        "median_time": statistics.median(times),
                        "min_time": min(times),
                        "max_time": max(times),
                        "std_dev": statistics.stdev(times) if len(times) > 1 else 0
                    }
            
            # Metric summaries
            for metric_name, data_points in self.metrics_history.items():
                if data_points:
                    values = [value for _, value in data_points[-100:]]  # Last 100 points
                    if values:
                        summary["metric_summaries"][metric_name] = {
                            "current": values[-1],
                            "average": statistics.mean(values),
                            "median": statistics.median(values),
                            "min": min(values),
                            "max": max(values),
                            "trend": self._calculate_trend(values)
                        }
            
            return summary
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for metric values."""
        if len(values) < 5:
            return "insufficient_data"
        
        recent = statistics.mean(values[-5:])
        older = statistics.mean(values[-10:-5]) if len(values) >= 10 else statistics.mean(values[:-5])
        
        if recent > older * 1.05:
            return "increasing"
        elif recent < older * 0.95:
            return "decreasing"
        else:
            return "stable"


class PerformanceBenchmark:
    """Comprehensive performance benchmark suite."""
    
    def __init__(self):
        self.benchmark_results: List[BenchmarkResult] = []
        self.profiler_data = {}
        
    def run_synthesis_throughput_benchmark(self, circuit_sizes: List[int] = None) -> BenchmarkResult:
        """Benchmark synthesis throughput for different circuit sizes."""
        if circuit_sizes is None:
            circuit_sizes = [10, 25, 50, 100, 200]
        
        start_time = time.time()
        metrics = {
            "circuit_sizes": circuit_sizes,
            "synthesis_times": [],
            "throughput_rates": [],
            "memory_usage": []
        }
        
        from .photonic_mlir_bridge import SynthesisBridge, PhotonicCircuitBuilder
        bridge = SynthesisBridge(enable_optimization=True)
        
        for size in circuit_sizes:
            logger.info(f"Benchmarking synthesis for {size} components...")
            
            # Create test circuit
            builder = PhotonicCircuitBuilder(f"benchmark_{size}")
            component_ids = []
            
            for i in range(size):
                if i % 3 == 0:
                    comp_id = builder.add_waveguide(10.0, position=(i, 0))
                elif i % 3 == 1:
                    comp_id = builder.add_beam_splitter(0.5, position=(i, 5))
                else:
                    comp_id = builder.add_phase_shifter(1.57, position=(i, -5))
                component_ids.append(comp_id)
            
            # Add connections
            for i in range(size - 1):
                builder.connect(component_ids[i], component_ids[i + 1], loss_db=0.1)
            
            circuit = builder.build()
            
            # Measure synthesis
            memory_before = tracemalloc.get_traced_memory()[0]
            synthesis_start = time.time()
            
            result = bridge.synthesize_circuit(circuit)
            
            synthesis_time = time.time() - synthesis_start
            memory_after = tracemalloc.get_traced_memory()[0]
            
            # Calculate metrics
            throughput = size / synthesis_time if synthesis_time > 0 else 0
            memory_delta = memory_after - memory_before
            
            metrics["synthesis_times"].append(synthesis_time)
            metrics["throughput_rates"].append(throughput)
            metrics["memory_usage"].append(memory_delta)
        
        total_duration = time.time() - start_time
        
        # Calculate statistics
        statistics_data = {
            "avg_synthesis_time": statistics.mean(metrics["synthesis_times"]),
            "median_synthesis_time": statistics.median(metrics["synthesis_times"]),
            "max_throughput": max(metrics["throughput_rates"]),
            "avg_throughput": statistics.mean(metrics["throughput_rates"]),
            "total_memory_delta": sum(metrics["memory_usage"]),
            "avg_memory_per_component": statistics.mean([m/s for m, s in zip(metrics["memory_usage"], circuit_sizes)])
        }
        
        result = BenchmarkResult(
            benchmark_type=BenchmarkType.SYNTHESIS_THROUGHPUT,
            timestamp=start_time,
            duration=total_duration,
            metrics=metrics,
            statistics=statistics_data,
            metadata={"optimization_enabled": True}
        )
        
        self.benchmark_results.append(result)
        return result
    
    def run_validation_performance_benchmark(self) -> BenchmarkResult:
        """Benchmark validation system performance."""
        from .photonic_validation import validate_photonic_circuit, ValidationLevel
        from .photonic_mlir_bridge import create_simple_mzi_circuit
        
        start_time = time.time()
        metrics = {
            "validation_levels": [],
            "validation_times": [],
            "issues_found": [],
            "memory_usage": []
        }
        
        test_circuit = create_simple_mzi_circuit()
        validation_levels = [ValidationLevel.BASIC, ValidationLevel.STANDARD, 
                           ValidationLevel.STRICT, ValidationLevel.PARANOID]
        
        for level in validation_levels:
            logger.info(f"Benchmarking validation level: {level.value}")
            
            memory_before = tracemalloc.get_traced_memory()[0]
            validation_start = time.time()
            
            report = validate_photonic_circuit(test_circuit, level)
            
            validation_time = time.time() - validation_start
            memory_after = tracemalloc.get_traced_memory()[0]
            
            metrics["validation_levels"].append(level.value)
            metrics["validation_times"].append(validation_time)
            metrics["issues_found"].append(len(report.issues))
            metrics["memory_usage"].append(memory_after - memory_before)
        
        total_duration = time.time() - start_time
        
        statistics_data = {
            "avg_validation_time": statistics.mean(metrics["validation_times"]),
            "fastest_validation": min(metrics["validation_times"]),
            "slowest_validation": max(metrics["validation_times"]),
            "total_issues": sum(metrics["issues_found"]),
            "avg_memory_usage": statistics.mean(metrics["memory_usage"])
        }
        
        result = BenchmarkResult(
            benchmark_type=BenchmarkType.VALIDATION_PERFORMANCE,
            timestamp=start_time,
            duration=total_duration,
            metrics=metrics,
            statistics=statistics_data,
            metadata={"circuit_components": len(test_circuit.components)}
        )
        
        self.benchmark_results.append(result)
        return result
    
    def run_concurrent_processing_benchmark(self, worker_counts: List[int] = None) -> BenchmarkResult:
        """Benchmark concurrent processing performance."""
        if worker_counts is None:
            worker_counts = [1, 2, 4, 8]
        
        start_time = time.time()
        metrics = {
            "worker_counts": worker_counts,
            "processing_times": [],
            "throughput_rates": [],
            "cpu_usage": [],
            "memory_usage": []
        }
        
        from .photonic_mlir_bridge import create_simple_mzi_circuit, SynthesisBridge
        
        test_circuit = create_simple_mzi_circuit()
        task_count = 20
        
        for worker_count in worker_counts:
            logger.info(f"Benchmarking {worker_count} concurrent workers...")
            
            # Monitor resources
            cpu_before = psutil.cpu_percent(interval=1)
            memory_before = tracemalloc.get_traced_memory()[0]
            
            # Execute concurrent tasks
            processing_start = time.time()
            
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = []
                
                for i in range(task_count):
                    bridge = SynthesisBridge(enable_optimization=True)
                    future = executor.submit(bridge.synthesize_circuit, test_circuit)
                    futures.append(future)
                
                # Wait for completion
                for future in futures:
                    future.result()
            
            processing_time = time.time() - processing_start
            
            # Monitor resources after
            cpu_after = psutil.cpu_percent(interval=1)
            memory_after = tracemalloc.get_traced_memory()[0]
            
            # Calculate metrics
            throughput = task_count / processing_time if processing_time > 0 else 0
            
            metrics["processing_times"].append(processing_time)
            metrics["throughput_rates"].append(throughput)
            metrics["cpu_usage"].append(cpu_after - cpu_before)
            metrics["memory_usage"].append(memory_after - memory_before)
        
        total_duration = time.time() - start_time
        
        statistics_data = {
            "best_throughput": max(metrics["throughput_rates"]),
            "optimal_worker_count": worker_counts[metrics["throughput_rates"].index(max(metrics["throughput_rates"]))],
            "avg_processing_time": statistics.mean(metrics["processing_times"]),
            "avg_cpu_overhead": statistics.mean(metrics["cpu_usage"]),
            "avg_memory_overhead": statistics.mean(metrics["memory_usage"])
        }
        
        result = BenchmarkResult(
            benchmark_type=BenchmarkType.CONCURRENT_PROCESSING,
            timestamp=start_time,
            duration=total_duration,
            metrics=metrics,
            statistics=statistics_data,
            metadata={"tasks_per_test": task_count}
        )
        
        self.benchmark_results.append(result)
        return result
    
    def run_memory_usage_benchmark(self) -> BenchmarkResult:
        """Benchmark memory usage patterns."""
        start_time = time.time()
        
        # Track memory for different operations
        memory_snapshots = []
        
        from .photonic_mlir_bridge import create_simple_mzi_circuit, SynthesisBridge
        from .photonic_validation import validate_photonic_circuit
        
        # Baseline memory
        baseline_memory = tracemalloc.get_traced_memory()[0]
        memory_snapshots.append(("baseline", baseline_memory))
        
        # Circuit creation
        circuit = create_simple_mzi_circuit()
        circuit_memory = tracemalloc.get_traced_memory()[0]
        memory_snapshots.append(("circuit_creation", circuit_memory))
        
        # Synthesis
        bridge = SynthesisBridge()
        synthesis_result = bridge.synthesize_circuit(circuit)
        synthesis_memory = tracemalloc.get_traced_memory()[0]
        memory_snapshots.append(("synthesis", synthesis_memory))
        
        # Validation
        validation_report = validate_photonic_circuit(circuit)
        validation_memory = tracemalloc.get_traced_memory()[0]
        memory_snapshots.append(("validation", validation_memory))
        
        # Calculate deltas
        memory_deltas = []
        for i in range(1, len(memory_snapshots)):
            prev_memory = memory_snapshots[i-1][1]
            curr_memory = memory_snapshots[i][1]
            delta = curr_memory - prev_memory
            memory_deltas.append((memory_snapshots[i][0], delta))
        
        total_duration = time.time() - start_time
        
        metrics = {
            "memory_snapshots": [(name, mem / (1024*1024)) for name, mem in memory_snapshots],  # MB
            "memory_deltas": [(name, delta / (1024*1024)) for name, delta in memory_deltas],  # MB
            "peak_memory": max(mem for _, mem in memory_snapshots) / (1024*1024)  # MB
        }
        
        statistics_data = {
            "total_memory_usage": (memory_snapshots[-1][1] - memory_snapshots[0][1]) / (1024*1024),
            "largest_delta": max(delta for _, delta in memory_deltas) / (1024*1024),
            "average_delta": statistics.mean([delta for _, delta in memory_deltas]) / (1024*1024)
        }
        
        result = BenchmarkResult(
            benchmark_type=BenchmarkType.MEMORY_USAGE,
            timestamp=start_time,
            duration=total_duration,
            metrics=metrics,
            statistics=statistics_data,
            metadata={}
        )
        
        self.benchmark_results.append(result)
        return result
    
    def profile_function(self, func: Callable, *args, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """Profile a function execution."""
        profiler = cProfile.Profile()
        
        # Run with profiling
        profiler.enable()
        try:
            result = func(*args, **kwargs)
        finally:
            profiler.disable()
        
        # Analyze profile data
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        
        # Extract top functions
        top_functions = []
        for func_name, (cc, nc, tt, ct, callers) in stats.stats.items():
            if tt > 0.001:  # Only functions with significant time
                profile = PerformanceProfile(
                    function_name=f"{func_name[2]}",
                    total_time=tt,
                    cumulative_time=ct,
                    call_count=cc,
                    time_per_call=tt/cc if cc > 0 else 0,
                    filename=func_name[0],
                    line_number=func_name[1]
                )
                top_functions.append(profile)
        
        # Sort by total time
        top_functions.sort(key=lambda x: x.total_time, reverse=True)
        
        profile_data = {
            "total_time": stats.total_tt,
            "top_functions": [
                {
                    "function": prof.function_name,
                    "total_time": prof.total_time,
                    "cumulative_time": prof.cumulative_time,
                    "call_count": prof.call_count,
                    "time_per_call": prof.time_per_call
                }
                for prof in top_functions[:10]
            ]
        }
        
        return result, profile_data
    
    def run_comprehensive_benchmark_suite(self) -> Dict[str, BenchmarkResult]:
        """Run comprehensive benchmark suite."""
        logger.info("Starting comprehensive performance benchmark suite...")
        
        results = {}
        
        # Synthesis throughput
        logger.info("Running synthesis throughput benchmark...")
        results["synthesis_throughput"] = self.run_synthesis_throughput_benchmark()
        
        # Validation performance
        logger.info("Running validation performance benchmark...")
        results["validation_performance"] = self.run_validation_performance_benchmark()
        
        # Concurrent processing
        logger.info("Running concurrent processing benchmark...")
        results["concurrent_processing"] = self.run_concurrent_processing_benchmark()
        
        # Memory usage
        logger.info("Running memory usage benchmark...")
        results["memory_usage"] = self.run_memory_usage_benchmark()
        
        logger.info("Comprehensive benchmark suite completed")
        return results
    
    def export_benchmark_results(self, filepath: str = None) -> str:
        """Export benchmark results to JSON file."""
        if filepath is None:
            filepath = f"benchmark_results_{int(time.time())}.json"
        
        export_data = {
            "export_timestamp": time.time(),
            "total_benchmarks": len(self.benchmark_results),
            "results": [
                {
                    "benchmark_type": result.benchmark_type.value,
                    "timestamp": result.timestamp,
                    "duration": result.duration,
                    "metrics": result.metrics,
                    "statistics": result.statistics,
                    "metadata": result.metadata
                }
                for result in self.benchmark_results
            ]
        }
        
        Path(filepath).write_text(json.dumps(export_data, indent=2))
        logger.info(f"Benchmark results exported to {filepath}")
        return filepath


# Performance decorators
def performance_monitor(operation_name: str = None):
    """Decorator to monitor function performance."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            start_time = time.time()
            success = True
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                raise
            finally:
                duration = time.time() - start_time
                
                # Record performance
                global _performance_monitor
                if _performance_monitor is None:
                    _performance_monitor = PerformanceMonitor()
                
                _performance_monitor.record_operation(op_name, duration, success)
        
        return wrapper
    return decorator


def cache_performance_monitor(func):
    """Decorator to monitor cache performance."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # This would integrate with actual cache implementation
        result = func(*args, **kwargs)
        
        # Record cache hit/miss
        global _performance_monitor
        if _performance_monitor is None:
            _performance_monitor = PerformanceMonitor()
        
        # This is a placeholder - real implementation would detect cache hits/misses
        _performance_monitor.record_cache_operation(hit=True)
        
        return result
    return wrapper


# Global performance monitor
_performance_monitor = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get or create global performance monitor."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor


def start_performance_monitoring():
    """Start global performance monitoring."""
    monitor = get_performance_monitor()
    monitor.start_monitoring()


def stop_performance_monitoring():
    """Stop global performance monitoring."""
    global _performance_monitor
    if _performance_monitor:
        _performance_monitor.stop_monitoring()


def get_performance_summary() -> Dict[str, Any]:
    """Get global performance summary."""
    global _performance_monitor
    if _performance_monitor is None:
        return {"performance_monitor": "not_initialized"}
    
    return _performance_monitor.get_performance_summary()


if __name__ == "__main__":
    # Demo performance suite
    print("⚡ Photonic-MLIR Bridge - Performance Suite Demo")
    print("=" * 60)
    
    # Start monitoring
    start_performance_monitoring()
    
    # Run benchmarks
    benchmark = PerformanceBenchmark()
    
    # Quick synthesis benchmark
    logger.info("Running quick synthesis benchmark...")
    synthesis_result = benchmark.run_synthesis_throughput_benchmark([10, 25])
    
    print(f"\nSynthesis Throughput Benchmark:")
    print(f"Average Time: {synthesis_result.statistics['avg_synthesis_time']:.3f}s")
    print(f"Max Throughput: {synthesis_result.statistics['max_throughput']:.1f} components/s")
    
    # Get performance summary
    summary = get_performance_summary()
    print(f"\nPerformance Summary:")
    print(f"Monitoring Active: {summary['monitoring_active']}")
    print(f"Peak Memory: {summary['peak_memory_mb']:.1f} MB")
    print(f"Cache Hit Rate: {summary['cache_statistics']['hit_rate']:.1%}")
    
    # Stop monitoring
    stop_performance_monitoring()
    print("\n✅ Performance suite operational!")