"""Basic load testing functionality for Generation 1 implementation."""
import time
import concurrent.futures
import requests
import threading
from typing import Dict, List, Any
import statistics
import logging

logger = logging.getLogger(__name__)


def run_load_test(target_url: str = "http://localhost:5000", 
                 num_requests: int = 100, 
                 concurrent_users: int = 10) -> Dict[str, Any]:
    """
    Run a basic load test for Generation 1 (MAKE IT WORK).
    Simple implementation with essential metrics.
    """
    logger.info(f"Starting basic load test: {num_requests} requests, {concurrent_users} users")
    
    results = {
        'total_requests': num_requests,
        'concurrent_users': concurrent_users,
        'response_times': [],
        'success_count': 0,
        'error_count': 0,
        'errors': []
    }
    
    def make_request():
        """Make a single HTTP request and record metrics."""
        try:
            start_time = time.time()
            
            # Test basic health endpoint
            response = requests.get(f"{target_url}/", timeout=5)
            
            duration = (time.time() - start_time) * 1000  # Convert to ms
            results['response_times'].append(duration)
            
            if response.status_code == 200:
                results['success_count'] += 1
            else:
                results['error_count'] += 1
                results['errors'].append(f"HTTP {response.status_code}")
                
        except Exception as e:
            results['error_count'] += 1
            results['errors'].append(str(e))
            results['response_times'].append(5000)  # Timeout value
    
    # Execute requests with thread pool
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_users) as executor:
        futures = [executor.submit(make_request) for _ in range(num_requests)]
        concurrent.futures.wait(futures)
    
    total_duration = time.time() - start_time
    
    # Calculate basic metrics
    if results['response_times']:
        results['avg_response_time'] = statistics.mean(results['response_times'])
        results['median_response_time'] = statistics.median(results['response_times'])
        results['max_response_time'] = max(results['response_times'])
        results['min_response_time'] = min(results['response_times'])
    else:
        results['avg_response_time'] = 0
        results['median_response_time'] = 0
        results['max_response_time'] = 0
        results['min_response_time'] = 0
    
    results['total_duration'] = total_duration
    results['requests_per_second'] = num_requests / total_duration if total_duration > 0 else 0
    results['success_rate'] = (results['success_count'] / num_requests) * 100
    
    logger.info(f"Load test completed: {results['success_rate']:.1f}% success rate, "
               f"{results['avg_response_time']:.2f}ms avg response time")
    
    return results


def run_basic_stress_test() -> bool:
    """Run a basic stress test to validate system under load."""
    try:
        # Test with minimal load first
        basic_result = run_load_test(
            target_url="http://localhost:5000",
            num_requests=10,
            concurrent_users=2
        )
        
        # Simple pass/fail criteria for Generation 1
        if basic_result['success_rate'] >= 80 and basic_result['avg_response_time'] < 1000:
            logger.info("✅ Basic stress test passed")
            return True
        else:
            logger.warning("⚠️ Basic stress test failed - continuing anyway for Generation 1")
            return False
            
    except Exception as e:
        logger.error(f"❌ Stress test error: {e}")
        return False


if __name__ == "__main__":
    # Simple test when run directly
    print("Running basic load test...")
    result = run_load_test(num_requests=20, concurrent_users=3)
    print(f"Success rate: {result['success_rate']:.1f}%")
    print(f"Average response time: {result['avg_response_time']:.2f}ms")