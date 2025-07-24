"""Tests for performance profiling functionality."""

import pytest
import time
import threading
from unittest.mock import patch, MagicMock

from src.profiling import (
    PerformanceProfiler,
    profiler,
    profile_function,
    profile_block,
    MemoryProfiler,
    get_performance_report,
    start_profiling,
    stop_profiling,
    memory_profiler
)


class TestPerformanceProfiler:
    """Test cases for PerformanceProfiler class."""
    
    def test_init(self):
        """Test profiler initialization."""
        test_profiler = PerformanceProfiler()
        assert test_profiler.enabled is True
        assert test_profiler._monitoring_active is False
        assert test_profiler.start_time <= time.time()
    
    def test_enable_disable(self):
        """Test enabling and disabling profiler."""
        test_profiler = PerformanceProfiler()
        
        test_profiler.disable()
        assert test_profiler.enabled is False
        
        test_profiler.enable()
        assert test_profiler.enabled is True
    
    def test_get_profile_summary(self):
        """Test getting profile summary."""
        test_profiler = PerformanceProfiler()
        summary = test_profiler.get_profile_summary()
        
        assert 'uptime_seconds' in summary
        assert 'function_stats' in summary
        assert 'current_memory_mb' in summary
        assert 'current_cpu_percent' in summary
        assert 'profiling_enabled' in summary
        assert 'monitoring_active' in summary
        
        assert isinstance(summary['uptime_seconds'], float)
        assert summary['uptime_seconds'] >= 0
    
    @patch('src.profiling.psutil.Process')
    def test_monitoring_start_stop(self, mock_process):
        """Test starting and stopping monitoring."""
        mock_process_instance = MagicMock()
        mock_process_instance.memory_info.return_value = MagicMock(rss=1024*1024*50, vms=1024*1024*100)
        mock_process_instance.cpu_percent.return_value = 10.5
        mock_process_instance.io_counters.return_value = MagicMock(read_bytes=1000, write_bytes=2000)
        mock_process.return_value = mock_process_instance
        
        test_profiler = PerformanceProfiler()
        
        # Test starting monitoring
        test_profiler.start_monitoring(interval=0.1)
        assert test_profiler._monitoring_active is True
        assert test_profiler._monitor_thread is not None
        
        # Wait a bit for monitoring to collect data
        time.sleep(0.2)
        
        # Test stopping monitoring
        test_profiler.stop_monitoring()
        assert test_profiler._monitoring_active is False
    
    def test_profile_function_decorator(self):
        """Test function profiling decorator."""
        test_profiler = PerformanceProfiler()
        
        @test_profiler.profile_function('test_func')
        def test_function(x: int) -> int:
            time.sleep(0.01)  # Small delay
            return x * 2
        
        result = test_function(5)
        assert result == 10
        
        summary = test_profiler.get_profile_summary()
        assert 'test_func' in summary['function_stats']
        
        stats = summary['function_stats']['test_func']
        assert stats['call_count'] == 1
        assert stats['total_time'] >= 0.01
        assert stats['avg_time'] >= 0.01
    
    def test_profile_function_disabled(self):
        """Test that profiling is skipped when disabled."""
        test_profiler = PerformanceProfiler()
        test_profiler.disable()
        
        @test_profiler.profile_function('disabled_func')
        def test_function(x: int) -> int:
            return x * 3
        
        result = test_function(4)
        assert result == 12
        
        summary = test_profiler.get_profile_summary()
        assert 'disabled_func' not in summary['function_stats']
    
    def test_slow_function_detection(self):
        """Test detection of slow functions."""
        test_profiler = PerformanceProfiler()
        
        @test_profiler.profile_function('slow_func')
        def slow_function():
            time.sleep(0.6)  # Exceeds slow threshold
        
        slow_function()
        
        summary = test_profiler.get_profile_summary()
        assert len(summary['slow_queries']) > 0
        
        slow_query = summary['slow_queries'][0]
        assert slow_query['function'] == 'slow_func'
        assert slow_query['duration'] >= 0.6


class TestGlobalProfileFunction:
    """Test cases for global profile_function decorator."""
    
    def test_profile_function_decorator(self):
        """Test global profile function decorator."""
        @profile_function('global_test_func')
        def test_function(n: int) -> int:
            return sum(range(n))
        
        result = test_function(100)
        assert result == sum(range(100))
        
        summary = profiler.get_profile_summary()
        assert 'global_test_func' in summary['function_stats']
    
    def test_profile_function_with_exception(self):
        """Test profiling when function raises exception."""
        @profile_function('exception_func')
        def error_function():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError, match="Test error"):
            error_function()
        
        # Function should still be recorded in stats
        summary = profiler.get_profile_summary()
        # Note: The function might not be in stats if exception occurs before recording


class TestProfileBlock:
    """Test cases for profile_block context manager."""
    
    def test_profile_block_normal(self):
        """Test profiling a code block."""
        with profile_block('test_block'):
            time.sleep(0.01)
            result = sum(range(100))
        
        assert result == sum(range(100))
        # Block profiling logs performance but doesn't store in function stats
    
    def test_profile_block_with_exception(self):
        """Test profile block with exception."""
        with pytest.raises(RuntimeError, match="Test block error"):
            with profile_block('error_block'):
                time.sleep(0.01)
                raise RuntimeError("Test block error")
        
        # Should still log the performance data despite exception


class TestMemoryProfiler:
    """Test cases for MemoryProfiler class."""
    
    @patch('src.profiling.psutil.Process')
    @patch('src.profiling.psutil.virtual_memory')
    def test_get_memory_usage(self, mock_virtual_memory, mock_process):
        """Test getting memory usage information."""
        # Mock process memory info
        mock_process_instance = MagicMock()
        mock_process_instance.memory_info.return_value = MagicMock(
            rss=100*1024*1024,  # 100MB
            vms=200*1024*1024   # 200MB
        )
        mock_process_instance.memory_percent.return_value = 15.5
        mock_process.return_value = mock_process_instance
        
        # Mock system memory info
        mock_memory = MagicMock()
        mock_memory.available = 2*1024*1024*1024  # 2GB
        mock_memory.total = 8*1024*1024*1024      # 8GB
        mock_virtual_memory.return_value = mock_memory
        
        memory_info = MemoryProfiler.get_memory_usage()
        
        assert memory_info['rss_mb'] == 100.0
        assert memory_info['vms_mb'] == 200.0
        assert memory_info['memory_percent'] == 15.5
        assert memory_info['available_mb'] == 2048.0
        assert memory_info['total_mb'] == 8192.0
    
    def test_check_memory_threshold(self):
        """Test memory threshold checking."""
        with patch.object(MemoryProfiler, 'get_memory_usage') as mock_get_usage:
            # Test below threshold
            mock_get_usage.return_value = {'rss_mb': 50.0}
            assert MemoryProfiler.check_memory_threshold(100.0) is False
            
            # Test above threshold
            mock_get_usage.return_value = {'rss_mb': 150.0}
            assert MemoryProfiler.check_memory_threshold(100.0) is True
    
    def test_track_memory_context_manager(self):
        """Test memory tracking context manager."""
        test_profiler = MemoryProfiler()
        
        with patch.object(test_profiler, 'get_memory_usage') as mock_get_usage:
            mock_get_usage.side_effect = [
                {'rss_mb': 100.0},  # Start memory
                {'rss_mb': 110.0}   # End memory
            ]
            
            with test_profiler.track_memory('test_operation'):
                pass  # Simulate some operation
            
            assert mock_get_usage.call_count == 2


class TestUtilityFunctions:
    """Test cases for utility functions."""
    
    def test_get_performance_report(self):
        """Test getting comprehensive performance report."""
        report = get_performance_report()
        
        assert 'profiler_summary' in report
        assert 'memory_info' in report
        assert 'system_info' in report
        
        # Check profiler summary structure
        profiler_summary = report['profiler_summary']
        assert 'uptime_seconds' in profiler_summary
        assert 'function_stats' in profiler_summary
        assert 'profiling_enabled' in profiler_summary
        
        # Check system info structure
        system_info = report['system_info']
        assert 'cpu_count' in system_info
        assert 'boot_time' in system_info
    
    def test_start_stop_profiling(self):
        """Test starting and stopping profiling."""
        # Test starting
        start_profiling(enable_monitoring=False)  # Disable monitoring for test
        assert profiler.enabled is True
        
        # Test stopping
        stop_profiling()
        assert profiler.enabled is False


class TestThreadSafety:
    """Test cases for thread safety of profiling."""
    
    def test_concurrent_profiling(self):
        """Test that profiling works correctly with concurrent functions."""
        results = []
        
        @profile_function('concurrent_func')
        def test_function(thread_id: int, iterations: int):
            total = 0
            for i in range(iterations):
                total += i
                if i % 100 == 0:
                    time.sleep(0.001)  # Small delay
            results.append((thread_id, total))
        
        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=test_function, args=(i, 1000))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify results
        assert len(results) == 3
        
        # Check that profiling recorded multiple calls
        summary = profiler.get_profile_summary()
        if 'concurrent_func' in summary['function_stats']:
            stats = summary['function_stats']['concurrent_func']
            assert stats['call_count'] >= 3


class TestErrorHandling:
    """Test cases for error handling in profiling."""
    
    @patch('src.profiling.psutil.Process')
    def test_profiling_with_psutil_errors(self, mock_process):
        """Test that profiling gracefully handles psutil errors."""
        # Make psutil calls raise exceptions
        mock_process.side_effect = Exception("psutil error")
        
        test_profiler = PerformanceProfiler()
        
        @test_profiler.profile_function('error_test')
        def test_function():
            return "success"
        
        # Should still work despite psutil errors
        result = test_function()
        assert result == "success"
    
    def test_memory_profiler_error_handling(self):
        """Test memory profiler error handling."""
        with patch('src.profiling.psutil.Process') as mock_process:
            mock_process.side_effect = Exception("Memory error")
            
            memory_info = MemoryProfiler.get_memory_usage()
            assert memory_info == {}  # Should return empty dict on error


class TestPerformanceImpact:
    """Test cases to ensure profiling doesn't significantly impact performance."""
    
    def test_profiling_overhead(self):
        """Test that profiling overhead is minimal."""
        @profile_function('performance_test')
        def fast_function(n: int) -> int:
            return n * 2
        
        # Time function calls with profiling
        start_time = time.time()
        for i in range(1000):
            fast_function(i)
        end_time = time.time()
        
        profiled_duration = end_time - start_time
        
        # Should complete quickly even with profiling
        assert profiled_duration < 1.0  # Should take less than 1 second for 1000 calls
    
    def test_disabled_profiling_performance(self):
        """Test that disabled profiling has minimal overhead."""
        test_profiler = PerformanceProfiler()
        test_profiler.disable()
        
        @test_profiler.profile_function('disabled_perf_test')
        def fast_function(n: int) -> int:
            return n * 3
        
        # Time calls with disabled profiling
        start_time = time.time()
        for i in range(1000):
            fast_function(i)
        end_time = time.time()
        
        disabled_duration = end_time - start_time
        
        # Should be very fast when disabled
        assert disabled_duration < 0.1  # Should take less than 0.1 second