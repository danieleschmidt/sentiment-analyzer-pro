
import asyncio
import threading
import multiprocessing
import time
from typing import Any, Callable, List, Optional, Union, Dict, Awaitable
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import queue
import logging
from functools import wraps

class TaskResult:
    """Result of a task execution."""
    
    def __init__(self, task_id: str, result: Any = None, error: Exception = None, 
                 duration: float = 0.0):
        self.task_id = task_id
        self.result = result
        self.error = error
        self.duration = duration
        self.timestamp = time.time()
    
    @property
    def success(self) -> bool:
        return self.error is None

class TaskQueue:
    """Thread-safe task queue with priority support."""
    
    def __init__(self, max_size: int = 1000):
        self.queue = queue.PriorityQueue(maxsize=max_size)
        self.task_count = 0
        self._lock = threading.Lock()
    
    def put_task(self, func: Callable, args: tuple = (), kwargs: dict = None, 
                 priority: int = 0) -> str:
        """Add task to queue with priority (lower numbers = higher priority)."""
        if kwargs is None:
            kwargs = {}
        
        with self._lock:
            task_id = f"task_{self.task_count}"
            self.task_count += 1
        
        task_item = (priority, time.time(), task_id, func, args, kwargs)
        self.queue.put(task_item)
        return task_id
    
    def get_task(self, timeout: Optional[float] = None):
        """Get next task from queue."""
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None

class ThreadPoolManager:
    """Advanced thread pool manager with auto-scaling."""
    
    def __init__(self, min_workers: int = 2, max_workers: int = 20, 
                 scale_threshold: float = 0.8):
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.scale_threshold = scale_threshold
        
        self.executor = ThreadPoolExecutor(max_workers=min_workers)
        self.current_workers = min_workers
        self.active_tasks = 0
        self.completed_tasks = 0
        self.failed_tasks = 0
        
        self._lock = threading.Lock()
        self._last_scale_check = time.time()
    
    def submit_task(self, func: Callable, *args, **kwargs):
        """Submit task to thread pool."""
        with self._lock:
            self.active_tasks += 1
        
        def wrapped_func():
            try:
                start_time = time.time()
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                with self._lock:
                    self.active_tasks -= 1
                    self.completed_tasks += 1
                
                return TaskResult(f"thread_task_{self.completed_tasks}", 
                                result, None, duration)
            
            except Exception as e:
                with self._lock:
                    self.active_tasks -= 1
                    self.failed_tasks += 1
                
                return TaskResult(f"thread_task_{self.failed_tasks}", 
                                None, e, time.time() - start_time)
        
        future = self.executor.submit(wrapped_func)
        
        # Check if we need to scale
        self._check_scaling()
        
        return future
    
    def _check_scaling(self):
        """Check if thread pool needs scaling."""
        current_time = time.time()
        
        # Only check every 30 seconds
        if current_time - self._last_scale_check < 30:
            return
        
        with self._lock:
            if self.active_tasks == 0:
                return
            
            utilization = self.active_tasks / self.current_workers
            
            # Scale up if high utilization
            if (utilization > self.scale_threshold and 
                self.current_workers < self.max_workers):
                
                new_worker_count = min(self.current_workers + 2, self.max_workers)
                self._resize_pool(new_worker_count)
                self.current_workers = new_worker_count
            
            # Scale down if low utilization
            elif (utilization < 0.3 and 
                  self.current_workers > self.min_workers):
                
                new_worker_count = max(self.current_workers - 1, self.min_workers)
                self._resize_pool(new_worker_count)
                self.current_workers = new_worker_count
            
            self._last_scale_check = current_time
    
    def _resize_pool(self, new_size: int):
        """Resize thread pool."""
        self.executor.shutdown(wait=False)
        self.executor = ThreadPoolExecutor(max_workers=new_size)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get thread pool statistics."""
        with self._lock:
            return {
                "current_workers": self.current_workers,
                "active_tasks": self.active_tasks,
                "completed_tasks": self.completed_tasks,
                "failed_tasks": self.failed_tasks,
                "utilization": self.active_tasks / self.current_workers if self.current_workers > 0 else 0
            }

class AsyncTaskProcessor:
    """Asynchronous task processor."""
    
    def __init__(self, max_concurrent: int = 100):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.active_tasks = 0
        self.completed_tasks = 0
        self.failed_tasks = 0
        self._lock = asyncio.Lock()
    
    async def process_task(self, coro: Awaitable) -> TaskResult:
        """Process async task with concurrency control."""
        async with self.semaphore:
            async with self._lock:
                task_id = f"async_task_{self.active_tasks + self.completed_tasks + self.failed_tasks}"
                self.active_tasks += 1
            
            start_time = time.time()
            
            try:
                result = await coro
                duration = time.time() - start_time
                
                async with self._lock:
                    self.active_tasks -= 1
                    self.completed_tasks += 1
                
                return TaskResult(task_id, result, None, duration)
            
            except Exception as e:
                duration = time.time() - start_time
                
                async with self._lock:
                    self.active_tasks -= 1
                    self.failed_tasks += 1
                
                return TaskResult(task_id, None, e, duration)
    
    async def process_batch(self, coros: List[Awaitable]) -> List[TaskResult]:
        """Process batch of async tasks."""
        tasks = [self.process_task(coro) for coro in coros]
        return await asyncio.gather(*tasks)

def parallel_map(func: Callable, items: List[Any], max_workers: int = None) -> List[Any]:
    """Parallel map using thread pool."""
    if not items:
        return []
    
    max_workers = max_workers or min(len(items), multiprocessing.cpu_count())
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(func, item) for item in items]
        results = []
        
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                results.append(e)
        
        return results

def process_parallel(func: Callable, items: List[Any], max_workers: int = None) -> List[Any]:
    """Parallel processing using process pool."""
    if not items:
        return []
    
    max_workers = max_workers or min(len(items), multiprocessing.cpu_count())
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(func, item) for item in items]
        results = []
        
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                results.append(e)
        
        return results

# Global instances
thread_pool_manager = ThreadPoolManager()
async_processor = AsyncTaskProcessor()
