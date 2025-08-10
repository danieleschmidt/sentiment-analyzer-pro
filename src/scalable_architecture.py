"""
Scalable architecture components for sentiment analyzer
Generation 3: Make It Scale - Auto-scaling and load balancing
"""
import asyncio
import aiohttp
from aiohttp import web
import time
import logging
import json
from typing import Dict, Any, List, Optional, AsyncGenerator
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor
import weakref
import queue
import psutil
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class LoadLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ResourceMetrics:
    """System resource metrics"""
    cpu_percent: float
    memory_percent: float
    disk_usage_percent: float
    network_io: Dict[str, int]
    process_count: int
    timestamp: float
    
    def get_load_level(self) -> LoadLevel:
        """Determine load level based on metrics"""
        max_resource = max(self.cpu_percent, self.memory_percent)
        
        if max_resource >= 90:
            return LoadLevel.CRITICAL
        elif max_resource >= 70:
            return LoadLevel.HIGH
        elif max_resource >= 40:
            return LoadLevel.MEDIUM
        else:
            return LoadLevel.LOW

class ResourceMonitor:
    """Monitor system resources and performance"""
    
    def __init__(self, check_interval: float = 5.0):
        self.check_interval = check_interval
        self.metrics_history = []
        self.max_history = 1000
        self.running = False
        self.monitor_thread = None
        
        # Callbacks for resource events
        self.high_load_callbacks = []
        self.low_load_callbacks = []
    
    def start_monitoring(self):
        """Start resource monitoring"""
        if not self.running:
            self.running = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        logger.info("Resource monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Keep history bounded
                if len(self.metrics_history) > self.max_history:
                    self.metrics_history = self.metrics_history[-self.max_history//2:]
                
                # Trigger callbacks based on load level
                load_level = metrics.get_load_level()
                if load_level in [LoadLevel.HIGH, LoadLevel.CRITICAL]:
                    self._trigger_callbacks(self.high_load_callbacks, metrics)
                elif load_level == LoadLevel.LOW:
                    self._trigger_callbacks(self.low_load_callbacks, metrics)
                
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
                time.sleep(self.check_interval)
    
    def _collect_metrics(self) -> ResourceMetrics:
        """Collect current system metrics"""
        try:
            net_io = psutil.net_io_counters()
            disk_usage = psutil.disk_usage('/')
            
            return ResourceMetrics(
                cpu_percent=psutil.cpu_percent(interval=1),
                memory_percent=psutil.virtual_memory().percent,
                disk_usage_percent=disk_usage.percent,
                network_io={
                    'bytes_sent': net_io.bytes_sent,
                    'bytes_recv': net_io.bytes_recv
                },
                process_count=len(psutil.pids()),
                timestamp=time.time()
            )
        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")
            return ResourceMetrics(0, 0, 0, {}, 0, time.time())
    
    def _trigger_callbacks(self, callbacks: List, metrics: ResourceMetrics):
        """Trigger registered callbacks"""
        for callback in callbacks:
            try:
                callback(metrics)
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    def add_high_load_callback(self, callback):
        """Add callback for high load events"""
        self.high_load_callbacks.append(callback)
    
    def add_low_load_callback(self, callback):
        """Add callback for low load events"""
        self.low_load_callbacks.append(callback)
    
    def get_current_metrics(self) -> Optional[ResourceMetrics]:
        """Get most recent metrics"""
        return self.metrics_history[-1] if self.metrics_history else None
    
    def get_average_metrics(self, minutes: int = 5) -> Optional[ResourceMetrics]:
        """Get average metrics over time period"""
        cutoff_time = time.time() - (minutes * 60)
        recent_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return None
        
        avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
        avg_disk = sum(m.disk_usage_percent for m in recent_metrics) / len(recent_metrics)
        
        return ResourceMetrics(
            cpu_percent=avg_cpu,
            memory_percent=avg_memory,
            disk_usage_percent=avg_disk,
            network_io={},
            process_count=recent_metrics[-1].process_count,
            timestamp=time.time()
        )

class WorkerPool:
    """Dynamic worker pool with auto-scaling"""
    
    def __init__(self, 
                 min_workers: int = 2,
                 max_workers: int = 10,
                 scale_up_threshold: float = 80.0,
                 scale_down_threshold: float = 30.0):
        
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        
        self.workers = {}
        self.task_queue = queue.Queue()
        self.result_futures = {}
        
        self.running = False
        self.worker_counter = 0
        
        # Performance tracking
        self.tasks_processed = 0
        self.total_processing_time = 0.0
        self.worker_utilization = {}
        
        # Create initial workers
        for _ in range(self.min_workers):
            self._create_worker()
    
    def _create_worker(self) -> str:
        """Create a new worker thread"""
        worker_id = f"worker_{self.worker_counter}"
        self.worker_counter += 1
        
        worker_thread = threading.Thread(
            target=self._worker_loop,
            args=(worker_id,),
            daemon=True
        )
        
        self.workers[worker_id] = {
            'thread': worker_thread,
            'busy': False,
            'tasks_processed': 0,
            'total_time': 0.0,
            'created_at': time.time()
        }
        
        worker_thread.start()
        logger.info(f"Created worker: {worker_id}")
        
        return worker_id
    
    def _remove_worker(self, worker_id: str):
        """Remove a worker (graceful shutdown)"""
        if worker_id in self.workers:
            # Mark for shutdown (worker will exit when it sees this)
            self.workers[worker_id]['shutdown'] = True
            del self.workers[worker_id]
            logger.info(f"Removed worker: {worker_id}")
    
    def _worker_loop(self, worker_id: str):
        """Main worker loop"""
        while self.running and worker_id in self.workers:
            try:
                # Check if marked for shutdown
                if self.workers[worker_id].get('shutdown', False):
                    break
                
                # Get task from queue (with timeout)
                try:
                    task = self.task_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Process task
                self.workers[worker_id]['busy'] = True
                start_time = time.time()
                
                try:
                    func, args, kwargs, future = task
                    result = func(*args, **kwargs)
                    future.set_result(result)
                except Exception as e:
                    future.set_exception(e)
                
                # Update statistics
                processing_time = time.time() - start_time
                self.workers[worker_id]['tasks_processed'] += 1
                self.workers[worker_id]['total_time'] += processing_time
                self.workers[worker_id]['busy'] = False
                
                self.tasks_processed += 1
                self.total_processing_time += processing_time
                
                self.task_queue.task_done()
                
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
    
    def start(self):
        """Start the worker pool"""
        self.running = True
        
        # Start worker threads
        for worker_id, worker_info in self.workers.items():
            if not worker_info['thread'].is_alive():
                worker_info['thread'].start()
        
        logger.info(f"Worker pool started with {len(self.workers)} workers")
    
    def stop(self):
        """Stop the worker pool"""
        self.running = False
        
        # Wait for current tasks to complete
        self.task_queue.join()
        
        # Wait for worker threads
        for worker_info in self.workers.values():
            worker_info['thread'].join(timeout=5.0)
        
        logger.info("Worker pool stopped")
    
    def submit_task(self, func, *args, **kwargs):
        """Submit task to worker pool"""
        import concurrent.futures
        
        future = concurrent.futures.Future()
        task = (func, args, kwargs, future)
        
        self.task_queue.put(task)
        return future
    
    def auto_scale(self, metrics: ResourceMetrics):
        """Auto-scale workers based on metrics"""
        current_workers = len(self.workers)
        busy_workers = sum(1 for w in self.workers.values() if w['busy'])
        utilization = (busy_workers / current_workers * 100) if current_workers > 0 else 0
        
        # Scale up if high utilization or high CPU/memory
        if (utilization > self.scale_up_threshold or 
            metrics.cpu_percent > 80 or metrics.memory_percent > 80):
            
            if current_workers < self.max_workers:
                self._create_worker()
                logger.info(f"Scaled up to {len(self.workers)} workers (utilization: {utilization:.1f}%)")
        
        # Scale down if low utilization and low resource usage
        elif (utilization < self.scale_down_threshold and 
              metrics.cpu_percent < 50 and metrics.memory_percent < 50):
            
            if current_workers > self.min_workers:
                # Remove oldest idle worker
                idle_workers = [wid for wid, w in self.workers.items() if not w['busy']]
                if idle_workers:
                    oldest_worker = min(idle_workers, 
                                      key=lambda wid: self.workers[wid]['created_at'])
                    self._remove_worker(oldest_worker)
                    logger.info(f"Scaled down to {len(self.workers)} workers (utilization: {utilization:.1f}%)")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker pool statistics"""
        current_workers = len(self.workers)
        busy_workers = sum(1 for w in self.workers.values() if w['busy'])
        
        avg_processing_time = (self.total_processing_time / self.tasks_processed 
                             if self.tasks_processed > 0 else 0)
        
        return {
            'current_workers': current_workers,
            'busy_workers': busy_workers,
            'utilization_percent': (busy_workers / current_workers * 100) if current_workers > 0 else 0,
            'tasks_processed': self.tasks_processed,
            'avg_processing_time_ms': avg_processing_time * 1000,
            'tasks_per_second': (self.tasks_processed / self.total_processing_time 
                               if self.total_processing_time > 0 else 0),
            'queue_size': self.task_queue.qsize()
        }

class AsyncSentimentAPI:
    """Async high-performance API server"""
    
    def __init__(self, 
                 host: str = "0.0.0.0",
                 port: int = 8000,
                 workers: int = 4):
        
        self.host = host
        self.port = port
        self.app = web.Application()
        
        # Setup components
        self.resource_monitor = ResourceMonitor()
        self.worker_pool = WorkerPool(min_workers=2, max_workers=workers)
        
        # Performance tracking
        self.request_count = 0
        self.error_count = 0
        self.total_response_time = 0.0
        
        # Setup routes
        self._setup_routes()
        self._setup_middleware()
    
    def _setup_routes(self):
        """Setup API routes"""
        self.app.router.add_post('/predict', self.predict_handler)
        self.app.router.add_post('/predict/batch', self.batch_predict_handler)
        self.app.router.add_get('/health', self.health_handler)
        self.app.router.add_get('/metrics', self.metrics_handler)
        self.app.router.add_get('/stats', self.stats_handler)
    
    def _setup_middleware(self):
        """Setup middleware"""
        self.app.middlewares.append(self.request_middleware)
        self.app.middlewares.append(self.error_middleware)
    
    async def request_middleware(self, request, handler):
        """Request tracking middleware"""
        start_time = time.time()
        
        try:
            response = await handler(request)
            self.request_count += 1
            self.total_response_time += time.time() - start_time
            return response
        except Exception as e:
            self.error_count += 1
            raise
    
    async def error_middleware(self, request, handler):
        """Error handling middleware"""
        try:
            return await handler(request)
        except Exception as e:
            logger.error(f"Request error: {e}")
            return web.json_response(
                {'error': 'Internal server error', 'message': str(e)}, 
                status=500
            )
    
    async def predict_handler(self, request):
        """Handle single prediction request"""
        try:
            data = await request.json()
            text = data.get('text', '')
            
            if not text:
                return web.json_response({'error': 'Missing text field'}, status=400)
            
            # Submit to worker pool
            future = self.worker_pool.submit_task(self._predict_sync, text)
            result = await asyncio.wrap_future(future)
            
            return web.json_response(result)
            
        except json.JSONDecodeError:
            return web.json_response({'error': 'Invalid JSON'}, status=400)
        except Exception as e:
            return web.json_response({'error': str(e)}, status=500)
    
    async def batch_predict_handler(self, request):
        """Handle batch prediction request"""
        try:
            data = await request.json()
            texts = data.get('texts', [])
            
            if not texts or len(texts) > 1000:
                return web.json_response(
                    {'error': 'Invalid texts field (max 1000 items)'}, 
                    status=400
                )
            
            # Submit batch to worker pool
            future = self.worker_pool.submit_task(self._predict_batch_sync, texts)
            results = await asyncio.wrap_future(future)
            
            return web.json_response({'results': results})
            
        except json.JSONDecodeError:
            return web.json_response({'error': 'Invalid JSON'}, status=400)
        except Exception as e:
            return web.json_response({'error': str(e)}, status=500)
    
    async def health_handler(self, request):
        """Health check endpoint"""
        metrics = self.resource_monitor.get_current_metrics()
        
        health_status = "healthy"
        if metrics:
            load_level = metrics.get_load_level()
            if load_level == LoadLevel.CRITICAL:
                health_status = "critical"
            elif load_level == LoadLevel.HIGH:
                health_status = "degraded"
        
        return web.json_response({
            'status': health_status,
            'timestamp': time.time(),
            'metrics': asdict(metrics) if metrics else None
        })
    
    async def metrics_handler(self, request):
        """Metrics endpoint"""
        worker_stats = self.worker_pool.get_stats()
        resource_metrics = self.resource_monitor.get_current_metrics()
        
        return web.json_response({
            'workers': worker_stats,
            'resources': asdict(resource_metrics) if resource_metrics else None,
            'api_stats': {
                'requests': self.request_count,
                'errors': self.error_count,
                'avg_response_time_ms': (
                    self.total_response_time / self.request_count * 1000
                    if self.request_count > 0 else 0
                )
            }
        })
    
    async def stats_handler(self, request):
        """Statistics endpoint"""
        return await self.metrics_handler(request)
    
    def _predict_sync(self, text: str) -> Dict[str, Any]:
        """Synchronous prediction for worker pool"""
        from .performance_engine import HighPerformanceSentimentAnalyzer
        
        # Simple prediction logic (replace with actual model)
        positive_words = ['good', 'great', 'excellent', 'amazing', 'love', 'fantastic']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'worst', 'horrible']
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            sentiment = 'positive'
            confidence = min(0.95, 0.6 + pos_count * 0.1)
        elif neg_count > pos_count:
            sentiment = 'negative'
            confidence = min(0.95, 0.6 + neg_count * 0.1)
        else:
            sentiment = 'neutral'
            confidence = 0.5
        
        return {
            'text': text,
            'sentiment': sentiment,
            'confidence': round(confidence, 3),
            'processing_time_ms': 1.0  # Placeholder
        }
    
    def _predict_batch_sync(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Synchronous batch prediction for worker pool"""
        return [self._predict_sync(text) for text in texts]
    
    async def start_server(self):
        """Start the async server"""
        # Setup auto-scaling callback
        self.resource_monitor.add_high_load_callback(self.worker_pool.auto_scale)
        
        # Start components
        self.resource_monitor.start_monitoring()
        self.worker_pool.start()
        
        # Start web server
        runner = web.AppRunner(self.app)
        await runner.setup()
        
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        
        logger.info(f"Async API server started on {self.host}:{self.port}")
        
        return runner
    
    async def stop_server(self, runner):
        """Stop the async server"""
        await runner.cleanup()
        self.worker_pool.stop()
        self.resource_monitor.stop_monitoring()
        
        logger.info("Async API server stopped")

class LoadBalancer:
    """Simple round-robin load balancer"""
    
    def __init__(self, backend_urls: List[str]):
        self.backend_urls = backend_urls
        self.current_index = 0
        self.backend_status = {url: True for url in backend_urls}
        self.lock = threading.Lock()
    
    def get_next_backend(self) -> Optional[str]:
        """Get next available backend URL"""
        with self.lock:
            available_backends = [url for url, status in self.backend_status.items() if status]
            
            if not available_backends:
                return None
            
            backend = available_backends[self.current_index % len(available_backends)]
            self.current_index += 1
            
            return backend
    
    def mark_backend_down(self, url: str):
        """Mark backend as unavailable"""
        with self.lock:
            self.backend_status[url] = False
    
    def mark_backend_up(self, url: str):
        """Mark backend as available"""
        with self.lock:
            self.backend_status[url] = True
    
    async def health_check_loop(self):
        """Periodically check backend health"""
        while True:
            for url in self.backend_urls:
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(f"{url}/health", timeout=5) as response:
                            if response.status == 200:
                                self.mark_backend_up(url)
                            else:
                                self.mark_backend_down(url)
                except:
                    self.mark_backend_down(url)
            
            await asyncio.sleep(30)  # Check every 30 seconds

if __name__ == "__main__":
    async def main():
        # Test scalable architecture
        api = AsyncSentimentAPI(port=8080)
        
        print("🔧 Starting scalable sentiment API...")
        runner = await api.start_server()
        
        print("🌐 Server running on http://localhost:8080")
        print("📊 Metrics available at http://localhost:8080/metrics")
        print("❤️  Health check at http://localhost:8080/health")
        
        try:
            # Keep server running
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Shutting down server...")
            await api.stop_server(runner)
    
    asyncio.run(main())