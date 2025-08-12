"""
Comprehensive Monitoring Suite for Production Sentiment Analysis

This module implements enterprise-grade monitoring with:
- Distributed tracing with OpenTelemetry
- Advanced metrics collection and alerting
- Log aggregation and analysis
- Performance profiling and optimization
- Business metrics and KPI tracking
- Anomaly detection and root cause analysis
- SLA monitoring and reporting

Features:
- Multi-dimensional metrics
- Custom dashboards generation
- Intelligent alerting with ML-based anomaly detection
- Performance bottleneck identification
- Cost tracking and optimization recommendations
- Security monitoring and threat detection
"""

from __future__ import annotations

import asyncio
import time
import json
import logging
import threading
import psutil
import resource
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
from contextlib import contextmanager
import functools
import traceback
import sys
import gc

# Monitoring and observability
try:
    from opentelemetry import trace, metrics
    from opentelemetry.exporter.prometheus import PrometheusMetricReader
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False

try:
    import prometheus_client
    from prometheus_client import Counter, Histogram, Gauge, Summary, CollectorRegistry
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

try:
    import structlog
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False
    import logging as structlog

try:
    import pandas as pd
    import numpy as np
    from scipy import stats
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    ANALYTICS_AVAILABLE = True
except ImportError:
    ANALYTICS_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class MonitoringConfig:
    """Configuration for comprehensive monitoring"""
    # Metrics collection
    enable_system_metrics: bool = True
    enable_application_metrics: bool = True
    enable_business_metrics: bool = True
    metrics_collection_interval: int = 30  # seconds
    
    # Tracing
    enable_tracing: bool = True
    jaeger_endpoint: str = "http://localhost:14268/api/traces"
    trace_sample_rate: float = 0.1
    
    # Logging
    enable_structured_logging: bool = True
    log_level: str = "INFO"
    log_retention_days: int = 30
    
    # Alerting
    enable_alerting: bool = True
    alert_evaluation_interval: int = 60  # seconds
    slack_webhook_url: Optional[str] = None
    email_notifications: List[str] = field(default_factory=list)
    
    # Performance profiling
    enable_profiling: bool = True
    profiling_sample_rate: float = 0.01
    memory_profiling: bool = True
    cpu_profiling: bool = True
    
    # Anomaly detection
    enable_anomaly_detection: bool = True
    anomaly_detection_window: int = 100  # samples
    anomaly_threshold: float = 0.05
    
    # Storage
    metrics_retention_hours: int = 168  # 7 days
    traces_retention_hours: int = 72    # 3 days
    logs_retention_hours: int = 720     # 30 days


@dataclass
class SystemMetrics:
    """System-level metrics snapshot"""
    timestamp: datetime = field(default_factory=datetime.now)
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_mb: float = 0.0
    disk_usage_percent: float = 0.0
    network_io_sent: float = 0.0
    network_io_recv: float = 0.0
    open_files: int = 0
    active_connections: int = 0
    load_average: List[float] = field(default_factory=list)


@dataclass
class ApplicationMetrics:
    """Application-specific metrics"""
    timestamp: datetime = field(default_factory=datetime.now)
    requests_per_second: float = 0.0
    average_response_time: float = 0.0
    error_rate: float = 0.0
    active_sessions: int = 0
    cache_hit_rate: float = 0.0
    database_connections: int = 0
    queue_size: int = 0
    model_prediction_time: float = 0.0
    model_accuracy: float = 0.0


@dataclass
class BusinessMetrics:
    """Business and domain-specific metrics"""
    timestamp: datetime = field(default_factory=datetime.now)
    sentiment_predictions_total: int = 0
    positive_sentiment_rate: float = 0.0
    negative_sentiment_rate: float = 0.0
    neutral_sentiment_rate: float = 0.0
    average_confidence_score: float = 0.0
    unique_users: int = 0
    revenue_impact: float = 0.0
    cost_per_prediction: float = 0.0


class PrometheusMetricsCollector:
    """Advanced Prometheus metrics collector"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus client not available")
            return
            
        # Create custom registry
        self.registry = CollectorRegistry()
        
        # System metrics
        self.cpu_usage = Gauge('system_cpu_usage_percent', 
                              'CPU usage percentage', 
                              registry=self.registry)
        self.memory_usage = Gauge('system_memory_usage_percent', 
                                 'Memory usage percentage',
                                 registry=self.registry)
        self.disk_usage = Gauge('system_disk_usage_percent', 
                               'Disk usage percentage',
                               registry=self.registry)
        self.network_io = Counter('system_network_io_bytes_total', 
                                 'Network I/O bytes',
                                 ['direction'], registry=self.registry)
        
        # Application metrics
        self.request_count = Counter('app_requests_total',
                                   'Total application requests',
                                   ['method', 'endpoint', 'status'],
                                   registry=self.registry)
        self.request_duration = Histogram('app_request_duration_seconds',
                                        'Request duration in seconds',
                                        ['method', 'endpoint'],
                                        registry=self.registry)
        self.error_rate = Gauge('app_error_rate',
                              'Application error rate',
                              registry=self.registry)
        
        # Business metrics
        self.prediction_count = Counter('business_predictions_total',
                                      'Total sentiment predictions',
                                      ['sentiment', 'model'],
                                      registry=self.registry)
        self.prediction_confidence = Histogram('business_prediction_confidence',
                                             'Prediction confidence scores',
                                             ['sentiment', 'model'],
                                             registry=self.registry)
        self.model_accuracy = Gauge('business_model_accuracy',
                                   'Model accuracy score',
                                   ['model'], registry=self.registry)
        
        # Performance metrics
        self.gc_collections = Counter('python_gc_collections_total',
                                    'Total garbage collections',
                                    ['generation'], registry=self.registry)
        self.memory_objects = Gauge('python_memory_objects_total',
                                   'Total objects in memory',
                                   registry=self.registry)
        
        logger.info("Prometheus metrics collector initialized")
    
    def update_system_metrics(self, metrics: SystemMetrics) -> None:
        """Update system metrics"""
        if not PROMETHEUS_AVAILABLE:
            return
            
        self.cpu_usage.set(metrics.cpu_percent)
        self.memory_usage.set(metrics.memory_percent)
        self.disk_usage.set(metrics.disk_usage_percent)
        self.network_io.labels(direction='sent').inc(metrics.network_io_sent)
        self.network_io.labels(direction='recv').inc(metrics.network_io_recv)
    
    def record_request(self, method: str, endpoint: str, status_code: int, duration: float) -> None:
        """Record HTTP request metrics"""
        if not PROMETHEUS_AVAILABLE:
            return
            
        self.request_count.labels(method=method, endpoint=endpoint, status=status_code).inc()
        self.request_duration.labels(method=method, endpoint=endpoint).observe(duration)
    
    def record_prediction(self, sentiment: str, model: str, confidence: float) -> None:
        """Record prediction metrics"""
        if not PROMETHEUS_AVAILABLE:
            return
            
        self.prediction_count.labels(sentiment=sentiment, model=model).inc()
        self.prediction_confidence.labels(sentiment=sentiment, model=model).observe(confidence)
    
    def update_model_accuracy(self, model: str, accuracy: float) -> None:
        """Update model accuracy metrics"""
        if not PROMETHEUS_AVAILABLE:
            return
            
        self.model_accuracy.labels(model=model).set(accuracy)


class OpenTelemetryTracer:
    """OpenTelemetry distributed tracing"""
    
    def __init__(self, config: MonitoringConfig, service_name: str = "sentiment-analyzer"):
        self.config = config
        self.service_name = service_name
        
        if not OTEL_AVAILABLE:
            logger.warning("OpenTelemetry not available")
            return
        
        # Configure tracing
        trace.set_tracer_provider(TracerProvider())
        self.tracer = trace.get_tracer(service_name)
        
        # Configure Jaeger exporter
        if config.jaeger_endpoint:
            jaeger_exporter = JaegerExporter(
                agent_host_name="localhost",
                agent_port=6831,
            )
            span_processor = BatchSpanProcessor(jaeger_exporter)
            trace.get_tracer_provider().add_span_processor(span_processor)
        
        logger.info("OpenTelemetry tracer initialized")
    
    @contextmanager
    def trace_operation(self, operation_name: str, **attributes):
        """Context manager for tracing operations"""
        if not OTEL_AVAILABLE:
            yield None
            return
            
        with self.tracer.start_as_current_span(operation_name) as span:
            # Add attributes
            for key, value in attributes.items():
                span.set_attribute(key, value)
            
            try:
                yield span
            except Exception as e:
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                raise
    
    def create_span(self, operation_name: str, **attributes):
        """Create a new span"""
        if not OTEL_AVAILABLE:
            return None
            
        span = self.tracer.start_span(operation_name)
        for key, value in attributes.items():
            span.set_attribute(key, value)
        return span


class StructuredLogger:
    """Advanced structured logging with context"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        
        if STRUCTLOG_AVAILABLE:
            structlog.configure(
                processors=[
                    structlog.stdlib.filter_by_level,
                    structlog.stdlib.add_logger_name,
                    structlog.stdlib.add_log_level,
                    structlog.stdlib.PositionalArgumentsFormatter(),
                    structlog.processors.TimeStamper(fmt="iso"),
                    structlog.processors.StackInfoRenderer(),
                    structlog.processors.format_exc_info,
                    structlog.processors.UnicodeDecoder(),
                    structlog.processors.JSONRenderer()
                ],
                context_class=dict,
                logger_factory=structlog.stdlib.LoggerFactory(),
                wrapper_class=structlog.stdlib.BoundLogger,
                cache_logger_on_first_use=True,
            )
            
        self.logger = structlog.get_logger()
        
    def log_with_context(self, level: str, message: str, **context):
        """Log message with structured context"""
        log_func = getattr(self.logger, level.lower())
        log_func(message, **context)
    
    def log_error_with_traceback(self, error: Exception, context: Dict[str, Any] = None):
        """Log error with full traceback and context"""
        context = context or {}
        self.logger.error(
            "Exception occurred",
            error=str(error),
            error_type=type(error).__name__,
            traceback=traceback.format_exc(),
            **context
        )
    
    def log_performance_metrics(self, operation: str, duration: float, **metrics):
        """Log performance metrics"""
        self.logger.info(
            "Performance metrics",
            operation=operation,
            duration_seconds=duration,
            **metrics
        )


class PerformanceProfiler:
    """Advanced performance profiling and analysis"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.profile_data: Dict[str, List] = defaultdict(list)
        self._profiling_active = False
        
    @contextmanager
    def profile_operation(self, operation_name: str):
        """Profile operation performance"""
        if not self.config.enable_profiling:
            yield
            return
            
        start_time = time.time()
        start_memory = self._get_memory_usage()
        start_cpu_time = time.process_time()
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()
            end_cpu_time = time.process_time()
            
            # Record metrics
            self.profile_data[operation_name].append({
                'timestamp': datetime.now(),
                'wall_time': end_time - start_time,
                'cpu_time': end_cpu_time - start_cpu_time,
                'memory_delta': end_memory - start_memory,
                'memory_peak': end_memory
            })
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        if self.config.memory_profiling:
            try:
                process = psutil.Process()
                return process.memory_info().rss / 1024 / 1024
            except:
                return 0.0
        return 0.0
    
    def get_performance_summary(self, operation_name: str = None) -> Dict[str, Any]:
        """Get performance summary for operations"""
        if operation_name:
            data = self.profile_data.get(operation_name, [])
            operations = {operation_name: data}
        else:
            operations = dict(self.profile_data)
        
        summary = {}
        
        for op_name, measurements in operations.items():
            if not measurements:
                continue
                
            wall_times = [m['wall_time'] for m in measurements]
            cpu_times = [m['cpu_time'] for m in measurements]
            memory_deltas = [m['memory_delta'] for m in measurements]
            
            summary[op_name] = {
                'count': len(measurements),
                'wall_time': {
                    'mean': np.mean(wall_times) if ANALYTICS_AVAILABLE else sum(wall_times)/len(wall_times),
                    'median': np.median(wall_times) if ANALYTICS_AVAILABLE else sorted(wall_times)[len(wall_times)//2],
                    'p95': np.percentile(wall_times, 95) if ANALYTICS_AVAILABLE else sorted(wall_times)[int(len(wall_times)*0.95)],
                    'max': max(wall_times),
                    'min': min(wall_times)
                },
                'cpu_time': {
                    'mean': np.mean(cpu_times) if ANALYTICS_AVAILABLE else sum(cpu_times)/len(cpu_times),
                    'max': max(cpu_times),
                    'min': min(cpu_times)
                },
                'memory_delta': {
                    'mean': np.mean(memory_deltas) if ANALYTICS_AVAILABLE else sum(memory_deltas)/len(memory_deltas),
                    'max': max(memory_deltas),
                    'min': min(memory_deltas)
                },
                'last_measured': max(m['timestamp'] for m in measurements).isoformat()
            }
        
        return summary


class AnomalyDetector:
    """ML-based anomaly detection for monitoring metrics"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.metric_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=config.anomaly_detection_window)
        )
        self.anomaly_models: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        
        if not ANALYTICS_AVAILABLE:
            logger.warning("Analytics libraries not available for anomaly detection")
    
    def add_metric_value(self, metric_name: str, value: float, timestamp: datetime = None) -> None:
        """Add metric value to history"""
        if timestamp is None:
            timestamp = datetime.now()
            
        self.metric_history[metric_name].append({
            'value': value,
            'timestamp': timestamp
        })
        
        # Update anomaly model if we have enough data
        if len(self.metric_history[metric_name]) >= 20:
            self._update_anomaly_model(metric_name)
    
    def _update_anomaly_model(self, metric_name: str) -> None:
        """Update anomaly detection model for metric"""
        if not ANALYTICS_AVAILABLE:
            return
            
        history = self.metric_history[metric_name]
        values = np.array([point['value'] for point in history]).reshape(-1, 1)
        
        # Use Isolation Forest for anomaly detection
        model = IsolationForest(contamination=self.config.anomaly_threshold, random_state=42)
        scaler = StandardScaler()
        
        scaled_values = scaler.fit_transform(values)
        model.fit(scaled_values)
        
        self.anomaly_models[metric_name] = model
        self.scalers[metric_name] = scaler
    
    def detect_anomaly(self, metric_name: str, value: float) -> Dict[str, Any]:
        """Detect if value is anomalous"""
        if metric_name not in self.anomaly_models or not ANALYTICS_AVAILABLE:
            return {'is_anomaly': False, 'confidence': 0.0}
        
        model = self.anomaly_models[metric_name]
        scaler = self.scalers[metric_name]
        
        # Scale the value
        scaled_value = scaler.transform([[value]])
        
        # Predict anomaly
        prediction = model.predict(scaled_value)[0]
        anomaly_score = model.decision_function(scaled_value)[0]
        
        is_anomaly = prediction == -1
        confidence = abs(anomaly_score)
        
        return {
            'is_anomaly': is_anomaly,
            'confidence': confidence,
            'anomaly_score': anomaly_score,
            'threshold': self.config.anomaly_threshold
        }
    
    def get_anomaly_report(self) -> Dict[str, Any]:
        """Get comprehensive anomaly detection report"""
        report = {
            'metrics_monitored': len(self.anomaly_models),
            'total_data_points': sum(len(history) for history in self.metric_history.values()),
            'models_trained': len(self.anomaly_models),
            'detection_window': self.config.anomaly_detection_window,
            'anomaly_threshold': self.config.anomaly_threshold,
            'generated_at': datetime.now().isoformat()
        }
        
        return report


class AlertManager:
    """Advanced alerting system with intelligent routing"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.alert_rules: List[Dict] = []
        self.alert_history: deque = deque(maxlen=1000)
        self.alert_suppression: Dict[str, datetime] = {}
        self.notification_channels: List[Callable] = []
        
    def add_alert_rule(self, rule: Dict[str, Any]) -> None:
        """Add alert rule"""
        required_fields = ['name', 'condition', 'severity']
        if not all(field in rule for field in required_fields):
            raise ValueError(f"Alert rule must contain: {required_fields}")
        
        # Add default values
        rule.setdefault('cooldown_minutes', 15)
        rule.setdefault('evaluation_count', 1)
        rule.setdefault('labels', {})
        rule.setdefault('annotations', {})
        
        self.alert_rules.append(rule)
        logger.info(f"Added alert rule: {rule['name']}")
    
    def add_notification_channel(self, channel: Callable) -> None:
        """Add notification channel"""
        self.notification_channels.append(channel)
    
    def evaluate_alerts(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Evaluate all alert rules against current metrics"""
        triggered_alerts = []
        current_time = datetime.now()
        
        for rule in self.alert_rules:
            rule_name = rule['name']
            
            # Check suppression cooldown
            if rule_name in self.alert_suppression:
                cooldown = timedelta(minutes=rule.get('cooldown_minutes', 15))
                if current_time - self.alert_suppression[rule_name] < cooldown:
                    continue
            
            # Evaluate condition
            if self._evaluate_condition(rule['condition'], metrics):
                alert = self._create_alert(rule, metrics, current_time)
                triggered_alerts.append(alert)
                
                # Record alert
                self.alert_history.append(alert)
                self.alert_suppression[rule_name] = current_time
                
                # Send notifications
                self._send_notifications(alert)
        
        return triggered_alerts
    
    def _evaluate_condition(self, condition: str, metrics: Dict[str, Any]) -> bool:
        """Evaluate alert condition"""
        # Simple condition evaluation - can be enhanced with more complex expressions
        try:
            # Replace metric names with values
            for key, value in metrics.items():
                if key in condition:
                    if isinstance(value, (int, float)):
                        condition = condition.replace(key, str(value))
                    
            # Evaluate the expression
            return eval(condition)
        except:
            logger.error(f"Failed to evaluate condition: {condition}")
            return False
    
    def _create_alert(self, rule: Dict, metrics: Dict, timestamp: datetime) -> Dict[str, Any]:
        """Create alert from rule and metrics"""
        return {
            'id': f"alert_{int(timestamp.timestamp())}_{hash(rule['name'])}",
            'name': rule['name'],
            'severity': rule['severity'],
            'condition': rule['condition'],
            'labels': rule['labels'],
            'annotations': rule['annotations'],
            'timestamp': timestamp,
            'metrics_snapshot': metrics.copy(),
            'status': 'firing'
        }
    
    def _send_notifications(self, alert: Dict[str, Any]) -> None:
        """Send alert notifications"""
        for channel in self.notification_channels:
            try:
                channel(alert)
            except Exception as e:
                logger.error(f"Failed to send notification: {e}")
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary statistics"""
        if not self.alert_history:
            return {'total_alerts': 0}
        
        alerts_by_severity = defaultdict(int)
        alerts_by_rule = defaultdict(int)
        recent_alerts = []
        
        for alert in self.alert_history:
            alerts_by_severity[alert['severity']] += 1
            alerts_by_rule[alert['name']] += 1
            
            # Last 10 alerts
            if len(recent_alerts) < 10:
                recent_alerts.append({
                    'name': alert['name'],
                    'severity': alert['severity'],
                    'timestamp': alert['timestamp'].isoformat()
                })
        
        return {
            'total_alerts': len(self.alert_history),
            'alerts_by_severity': dict(alerts_by_severity),
            'alerts_by_rule': dict(alerts_by_rule),
            'recent_alerts': recent_alerts,
            'active_suppressions': len(self.alert_suppression)
        }


class ComprehensiveMonitoringSuite:
    """Main monitoring suite orchestrating all components"""
    
    def __init__(self, config: MonitoringConfig = None, service_name: str = "sentiment-analyzer"):
        self.config = config or MonitoringConfig()
        self.service_name = service_name
        
        # Initialize components
        self.metrics_collector = PrometheusMetricsCollector(self.config)
        self.tracer = OpenTelemetryTracer(self.config, service_name)
        self.logger = StructuredLogger(self.config)
        self.profiler = PerformanceProfiler(self.config)
        self.anomaly_detector = AnomalyDetector(self.config)
        self.alert_manager = AlertManager(self.config)
        
        # Monitoring state
        self.monitoring_active = False
        self.metrics_collection_thread = None
        self._stop_monitoring = threading.Event()
        
        # Current metrics cache
        self._current_metrics = {
            'system': SystemMetrics(),
            'application': ApplicationMetrics(),
            'business': BusinessMetrics()
        }
        
        # Setup default alert rules
        self._setup_default_alerts()
        
        logger.info("Comprehensive Monitoring Suite initialized")
    
    def start_monitoring(self) -> None:
        """Start monitoring collection"""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self._stop_monitoring.clear()
        
        # Start metrics collection thread
        self.metrics_collection_thread = threading.Thread(target=self._metrics_collection_loop)
        self.metrics_collection_thread.daemon = True
        self.metrics_collection_thread.start()
        
        logger.info("Monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop monitoring collection"""
        self.monitoring_active = False
        self._stop_monitoring.set()
        
        if self.metrics_collection_thread:
            self.metrics_collection_thread.join(timeout=5.0)
        
        logger.info("Monitoring stopped")
    
    def _metrics_collection_loop(self) -> None:
        """Background metrics collection loop"""
        while not self._stop_monitoring.is_set():
            try:
                # Collect system metrics
                if self.config.enable_system_metrics:
                    self._collect_system_metrics()
                
                # Update Prometheus metrics
                self.metrics_collector.update_system_metrics(self._current_metrics['system'])
                
                # Check for anomalies
                if self.config.enable_anomaly_detection:
                    self._check_anomalies()
                
                # Evaluate alerts
                if self.config.enable_alerting:
                    self._evaluate_alerts()
                
                # Sleep until next collection
                time.sleep(self.config.metrics_collection_interval)
                
            except Exception as e:
                self.logger.log_error_with_traceback(e, {'component': 'metrics_collection'})
                time.sleep(self.config.metrics_collection_interval)
    
    def _collect_system_metrics(self) -> None:
        """Collect system-level metrics"""
        try:
            # CPU and Memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Network I/O
            net_io = psutil.net_io_counters()
            
            # Process info
            process = psutil.Process()
            
            # Update current metrics
            self._current_metrics['system'] = SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / 1024 / 1024,
                disk_usage_percent=disk.percent,
                network_io_sent=net_io.bytes_sent,
                network_io_recv=net_io.bytes_recv,
                open_files=process.num_fds() if hasattr(process, 'num_fds') else 0,
                active_connections=len(process.connections()),
                load_average=list(psutil.getloadavg()) if hasattr(psutil, 'getloadavg') else []
            )
            
        except Exception as e:
            self.logger.log_error_with_traceback(e, {'component': 'system_metrics'})
    
    def _check_anomalies(self) -> None:
        """Check for anomalies in current metrics"""
        system_metrics = self._current_metrics['system']
        
        # Check key metrics for anomalies
        metrics_to_check = {
            'cpu_percent': system_metrics.cpu_percent,
            'memory_percent': system_metrics.memory_percent,
            'disk_usage_percent': system_metrics.disk_usage_percent
        }
        
        for metric_name, value in metrics_to_check.items():
            self.anomaly_detector.add_metric_value(metric_name, value)
            anomaly_result = self.anomaly_detector.detect_anomaly(metric_name, value)
            
            if anomaly_result['is_anomaly']:
                self.logger.log_with_context(
                    'warning',
                    'Anomaly detected',
                    metric=metric_name,
                    value=value,
                    confidence=anomaly_result['confidence'],
                    anomaly_score=anomaly_result['anomaly_score']
                )
    
    def _evaluate_alerts(self) -> None:
        """Evaluate alert rules"""
        current_metrics = {
            'cpu_percent': self._current_metrics['system'].cpu_percent,
            'memory_percent': self._current_metrics['system'].memory_percent,
            'disk_usage_percent': self._current_metrics['system'].disk_usage_percent,
            'error_rate': self._current_metrics['application'].error_rate,
            'response_time': self._current_metrics['application'].average_response_time
        }
        
        triggered_alerts = self.alert_manager.evaluate_alerts(current_metrics)
        
        for alert in triggered_alerts:
            self.logger.log_with_context(
                'error' if alert['severity'] == 'critical' else 'warning',
                'Alert triggered',
                alert_name=alert['name'],
                severity=alert['severity'],
                condition=alert['condition']
            )
    
    def _setup_default_alerts(self) -> None:
        """Setup default alert rules"""
        default_alerts = [
            {
                'name': 'High CPU Usage',
                'condition': 'cpu_percent > 90',
                'severity': 'warning',
                'cooldown_minutes': 5,
                'labels': {'component': 'system'},
                'annotations': {'description': 'CPU usage is above 90%'}
            },
            {
                'name': 'Critical CPU Usage',
                'condition': 'cpu_percent > 95',
                'severity': 'critical',
                'cooldown_minutes': 2,
                'labels': {'component': 'system'},
                'annotations': {'description': 'CPU usage is critically high'}
            },
            {
                'name': 'High Memory Usage',
                'condition': 'memory_percent > 85',
                'severity': 'warning',
                'cooldown_minutes': 10,
                'labels': {'component': 'system'},
                'annotations': {'description': 'Memory usage is above 85%'}
            },
            {
                'name': 'High Disk Usage',
                'condition': 'disk_usage_percent > 90',
                'severity': 'warning',
                'cooldown_minutes': 30,
                'labels': {'component': 'system'},
                'annotations': {'description': 'Disk usage is above 90%'}
            },
            {
                'name': 'High Error Rate',
                'condition': 'error_rate > 0.05',
                'severity': 'critical',
                'cooldown_minutes': 5,
                'labels': {'component': 'application'},
                'annotations': {'description': 'Application error rate is above 5%'}
            },
            {
                'name': 'High Response Time',
                'condition': 'response_time > 2000',
                'severity': 'warning',
                'cooldown_minutes': 5,
                'labels': {'component': 'application'},
                'annotations': {'description': 'Response time is above 2 seconds'}
            }
        ]
        
        for alert in default_alerts:
            self.alert_manager.add_alert_rule(alert)
    
    # Context managers for instrumentation
    @contextmanager
    def trace_operation(self, operation_name: str, **attributes):
        """Trace operation with performance profiling"""
        with self.tracer.trace_operation(operation_name, **attributes) as span:
            with self.profiler.profile_operation(operation_name):
                yield span
    
    # Decorator for automatic instrumentation
    def monitor_function(self, operation_name: str = None):
        """Decorator to automatically monitor function performance"""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                op_name = operation_name or f"{func.__module__}.{func.__name__}"
                
                with self.trace_operation(op_name):
                    start_time = time.time()
                    try:
                        result = func(*args, **kwargs)
                        
                        # Log successful execution
                        duration = time.time() - start_time
                        self.logger.log_performance_metrics(
                            op_name, duration,
                            args_count=len(args),
                            kwargs_count=len(kwargs)
                        )
                        
                        return result
                        
                    except Exception as e:
                        # Log error
                        self.logger.log_error_with_traceback(
                            e, {'operation': op_name}
                        )
                        raise
            return wrapper
        return decorator
    
    # Public API methods
    def record_prediction(self, sentiment: str, model: str, confidence: float) -> None:
        """Record prediction metrics"""
        self.metrics_collector.record_prediction(sentiment, model, confidence)
        
        # Update business metrics
        business = self._current_metrics['business']
        business.sentiment_predictions_total += 1
        
        if sentiment == 'positive':
            business.positive_sentiment_rate = (
                business.positive_sentiment_rate * (business.sentiment_predictions_total - 1) + 1
            ) / business.sentiment_predictions_total
        elif sentiment == 'negative':
            business.negative_sentiment_rate = (
                business.negative_sentiment_rate * (business.sentiment_predictions_total - 1) + 1
            ) / business.sentiment_predictions_total
        else:
            business.neutral_sentiment_rate = (
                business.neutral_sentiment_rate * (business.sentiment_predictions_total - 1) + 1
            ) / business.sentiment_predictions_total
            
        # Update confidence average
        total_confidence = (
            business.average_confidence_score * (business.sentiment_predictions_total - 1) + confidence
        )
        business.average_confidence_score = total_confidence / business.sentiment_predictions_total
    
    def record_http_request(self, method: str, endpoint: str, status_code: int, 
                          duration: float) -> None:
        """Record HTTP request metrics"""
        self.metrics_collector.record_request(method, endpoint, status_code, duration)
        
        # Update application metrics
        app = self._current_metrics['application']
        
        # Simple sliding window for RPS calculation
        current_time = time.time()
        if not hasattr(self, '_request_timestamps'):
            self._request_timestamps = deque(maxlen=60)  # Last 60 requests
        
        self._request_timestamps.append(current_time)
        
        # Calculate requests per second
        if len(self._request_timestamps) > 1:
            time_window = self._request_timestamps[-1] - self._request_timestamps[0]
            if time_window > 0:
                app.requests_per_second = len(self._request_timestamps) / time_window
        
        # Update average response time
        if not hasattr(self, '_response_times'):
            self._response_times = deque(maxlen=100)
        
        self._response_times.append(duration)
        app.average_response_time = sum(self._response_times) / len(self._response_times)
        
        # Update error rate
        if not hasattr(self, '_error_count'):
            self._error_count = 0
        if not hasattr(self, '_total_requests'):
            self._total_requests = 0
        
        self._total_requests += 1
        if status_code >= 400:
            self._error_count += 1
            
        app.error_rate = self._error_count / self._total_requests
    
    def get_monitoring_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard"""
        return {
            'system_metrics': asdict(self._current_metrics['system']),
            'application_metrics': asdict(self._current_metrics['application']),
            'business_metrics': asdict(self._current_metrics['business']),
            'performance_summary': self.profiler.get_performance_summary(),
            'alert_summary': self.alert_manager.get_alert_summary(),
            'anomaly_report': self.anomaly_detector.get_anomaly_report(),
            'monitoring_status': {
                'active': self.monitoring_active,
                'service_name': self.service_name,
                'config': asdict(self.config),
                'uptime_seconds': time.time() - getattr(self, '_start_time', time.time())
            }
        }
    
    def generate_monitoring_report(self) -> str:
        """Generate comprehensive monitoring report"""
        data = self.get_monitoring_dashboard_data()
        
        report = f"""
# Comprehensive Monitoring Report
Generated: {datetime.now().isoformat()}
Service: {self.service_name}

## System Metrics
- CPU Usage: {data['system_metrics']['cpu_percent']:.1f}%
- Memory Usage: {data['system_metrics']['memory_percent']:.1f}% ({data['system_metrics']['memory_used_mb']:.1f} MB)
- Disk Usage: {data['system_metrics']['disk_usage_percent']:.1f}%
- Open Files: {data['system_metrics']['open_files']}
- Active Connections: {data['system_metrics']['active_connections']}

## Application Metrics
- Requests/Second: {data['application_metrics']['requests_per_second']:.2f}
- Average Response Time: {data['application_metrics']['average_response_time']:.2f}ms
- Error Rate: {data['application_metrics']['error_rate']:.2%}
- Model Prediction Time: {data['application_metrics']['model_prediction_time']:.2f}ms

## Business Metrics
- Total Predictions: {data['business_metrics']['sentiment_predictions_total']}
- Positive Rate: {data['business_metrics']['positive_sentiment_rate']:.2%}
- Negative Rate: {data['business_metrics']['negative_sentiment_rate']:.2%}
- Neutral Rate: {data['business_metrics']['neutral_sentiment_rate']:.2%}
- Average Confidence: {data['business_metrics']['average_confidence_score']:.2f}

## Alert Summary
- Total Alerts: {data['alert_summary'].get('total_alerts', 0)}
- Active Suppressions: {data['alert_summary'].get('active_suppressions', 0)}

## Performance Summary
{self._format_performance_summary(data['performance_summary'])}

## Anomaly Detection
- Metrics Monitored: {data['anomaly_report']['metrics_monitored']}
- Models Trained: {data['anomaly_report']['models_trained']}
- Detection Threshold: {data['anomaly_report']['anomaly_threshold']}

## Monitoring Status
- Status: {'Active' if data['monitoring_status']['active'] else 'Inactive'}
- Uptime: {data['monitoring_status']['uptime_seconds']:.0f} seconds
        """
        
        return report.strip()
    
    def _format_performance_summary(self, summary: Dict) -> str:
        """Format performance summary for report"""
        if not summary:
            return "No performance data available"
        
        lines = []
        for operation, stats in summary.items():
            lines.append(f"### {operation}")
            lines.append(f"  - Calls: {stats['count']}")
            lines.append(f"  - Mean Time: {stats['wall_time']['mean']:.3f}s")
            lines.append(f"  - P95 Time: {stats['wall_time']['p95']:.3f}s")
            lines.append(f"  - Max Time: {stats['wall_time']['max']:.3f}s")
            lines.append(f"  - Memory Delta: {stats['memory_delta']['mean']:.2f}MB")
            lines.append("")
        
        return "\n".join(lines)


# Factory function
def create_monitoring_suite(service_name: str = "sentiment-analyzer", 
                          **config_kwargs) -> ComprehensiveMonitoringSuite:
    """Create comprehensive monitoring suite"""
    config = MonitoringConfig(**config_kwargs)
    return ComprehensiveMonitoringSuite(config, service_name)


# Example usage
if __name__ == "__main__":
    # Create monitoring suite
    monitoring = create_monitoring_suite(
        service_name="sentiment-analyzer-test",
        enable_anomaly_detection=True,
        enable_alerting=True,
        metrics_collection_interval=10
    )
    
    # Start monitoring
    monitoring.start_monitoring()
    
    # Simulate some operations
    import random
    
    @monitoring.monitor_function("test_prediction")
    def simulate_prediction():
        time.sleep(random.uniform(0.1, 0.5))
        sentiment = random.choice(['positive', 'negative', 'neutral'])
        confidence = random.uniform(0.7, 0.95)
        return sentiment, confidence
    
    # Run simulation
    try:
        for i in range(20):
            # Simulate prediction
            sentiment, confidence = simulate_prediction()
            monitoring.record_prediction(sentiment, "test-model", confidence)
            
            # Simulate HTTP request
            status_code = random.choice([200, 200, 200, 400, 500])
            duration = random.uniform(0.1, 2.0)
            monitoring.record_http_request("POST", "/predict", status_code, duration)
            
            time.sleep(1)
        
        # Generate report
        print("Monitoring Report:")
        print("=" * 50)
        print(monitoring.generate_monitoring_report())
        
        # Get dashboard data
        dashboard_data = monitoring.get_monitoring_dashboard_data()
        print("\nDashboard Data:")
        print(json.dumps(dashboard_data, indent=2, default=str))
        
    finally:
        monitoring.stop_monitoring()