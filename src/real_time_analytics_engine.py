"""
Real-Time Analytics Engine for Sentiment Analysis

This module implements high-performance real-time analytics capabilities:
- Stream processing for live sentiment analysis
- Real-time dashboards and metrics
- Event-driven architecture
- Temporal pattern detection
- Anomaly detection in sentiment streams
- Multi-dimensional analytics

Features:
- WebSocket streaming for live updates
- Time-series sentiment analytics
- Geospatial sentiment mapping
- Topic trend analysis
- Real-time model performance monitoring
- Custom alert system
"""

from __future__ import annotations

import asyncio
import websockets
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Callable, AsyncIterator
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import deque, defaultdict
import threading
import logging
from pathlib import Path
import uuid

# Time series and analytics
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Optional dependencies
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import kafka
    from kafka import KafkaConsumer, KafkaProducer
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class SentimentEvent:
    """Represents a single sentiment analysis event"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    text: str = ""
    sentiment: str = ""
    confidence: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "unknown"
    user_id: Optional[str] = None
    location: Optional[Dict[str, float]] = None  # {"lat": ..., "lon": ...}
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AnalyticsMetrics:
    """Container for real-time analytics metrics"""
    total_events: int = 0
    events_per_second: float = 0.0
    sentiment_distribution: Dict[str, int] = field(default_factory=dict)
    average_confidence: float = 0.0
    top_sources: Dict[str, int] = field(default_factory=dict)
    geographic_distribution: Dict[str, int] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class StreamProcessor:
    """High-performance stream processing for sentiment events"""
    
    def __init__(self, buffer_size: int = 10000, batch_size: int = 100):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.event_buffer: deque = deque(maxlen=buffer_size)
        self.processed_count = 0
        self.processing_callbacks: List[Callable] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._processing_thread = None
        
    def add_callback(self, callback: Callable[[List[SentimentEvent]], None]) -> None:
        """Add callback function to process batches of events"""
        self.processing_callbacks.append(callback)
        
    def push_event(self, event: SentimentEvent) -> None:
        """Add event to processing buffer"""
        with self._lock:
            self.event_buffer.append(event)
            self.processed_count += 1
            
    def start_processing(self) -> None:
        """Start background processing of events"""
        if self._processing_thread and self._processing_thread.is_alive():
            return
            
        self._stop_event.clear()
        self._processing_thread = threading.Thread(target=self._process_events)
        self._processing_thread.daemon = True
        self._processing_thread.start()
        logger.info("Stream processing started")
        
    def stop_processing(self) -> None:
        """Stop background processing"""
        self._stop_event.set()
        if self._processing_thread:
            self._processing_thread.join(timeout=5.0)
        logger.info("Stream processing stopped")
        
    def _process_events(self) -> None:
        """Background event processing loop"""
        while not self._stop_event.is_set():
            batch = []
            
            with self._lock:
                # Extract batch from buffer
                batch_size = min(self.batch_size, len(self.event_buffer))
                batch = [self.event_buffer.popleft() for _ in range(batch_size)]
            
            if batch:
                # Process batch with all callbacks
                for callback in self.processing_callbacks:
                    try:
                        callback(batch)
                    except Exception as e:
                        logger.error(f"Error in processing callback: {e}")
            else:
                # Sleep briefly if no events to process
                time.sleep(0.1)


class TimeSeriesAnalyzer:
    """Analyzes temporal patterns in sentiment streams"""
    
    def __init__(self, window_minutes: int = 60):
        self.window_minutes = window_minutes
        self.time_series_data: Dict[str, deque] = {
            'positive': deque(),
            'negative': deque(), 
            'neutral': deque()
        }
        self.anomaly_detector = None
        
    def add_sentiment_point(self, sentiment: str, timestamp: datetime = None) -> None:
        """Add sentiment data point to time series"""
        if timestamp is None:
            timestamp = datetime.now()
            
        # Add to appropriate series
        if sentiment in self.time_series_data:
            self.time_series_data[sentiment].append({
                'timestamp': timestamp,
                'value': 1
            })
            
        # Clean old data outside window
        cutoff_time = datetime.now() - timedelta(minutes=self.window_minutes)
        for sentiment_type in self.time_series_data:
            while (self.time_series_data[sentiment_type] and 
                   self.time_series_data[sentiment_type][0]['timestamp'] < cutoff_time):
                self.time_series_data[sentiment_type].popleft()
    
    def get_sentiment_trends(self, granularity_minutes: int = 5) -> Dict[str, List]:
        """Get sentiment trends aggregated by time granularity"""
        trends = {}
        
        for sentiment_type, data_points in self.time_series_data.items():
            if not data_points:
                trends[sentiment_type] = []
                continue
                
            # Group by time buckets
            bucket_counts = defaultdict(int)
            for point in data_points:
                # Round timestamp to nearest granularity
                bucket_time = self._round_to_granularity(
                    point['timestamp'], granularity_minutes
                )
                bucket_counts[bucket_time] += point['value']
            
            # Convert to sorted list
            trends[sentiment_type] = [
                {'timestamp': ts, 'count': count}
                for ts, count in sorted(bucket_counts.items())
            ]
            
        return trends
    
    def detect_anomalies(self, sentiment_type: str = 'all') -> List[Dict[str, Any]]:
        """Detect anomalies in sentiment patterns using statistical methods"""
        anomalies = []
        
        sentiment_types = [sentiment_type] if sentiment_type != 'all' else list(self.time_series_data.keys())
        
        for s_type in sentiment_types:
            data_points = list(self.time_series_data[s_type])
            if len(data_points) < 10:  # Need minimum data for detection
                continue
                
            # Extract values and timestamps
            values = [point['value'] for point in data_points]
            timestamps = [point['timestamp'] for point in data_points]
            
            # Simple z-score based anomaly detection
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            if std_val > 0:
                z_scores = np.abs((values - mean_val) / std_val)
                anomaly_threshold = 2.5
                
                for i, (z_score, timestamp) in enumerate(zip(z_scores, timestamps)):
                    if z_score > anomaly_threshold:
                        anomalies.append({
                            'sentiment_type': s_type,
                            'timestamp': timestamp,
                            'value': values[i],
                            'z_score': z_score,
                            'severity': 'high' if z_score > 3.0 else 'medium'
                        })
        
        return sorted(anomalies, key=lambda x: x['timestamp'], reverse=True)
    
    def _round_to_granularity(self, timestamp: datetime, granularity_minutes: int) -> datetime:
        """Round timestamp to specified granularity"""
        minutes = timestamp.minute
        rounded_minutes = (minutes // granularity_minutes) * granularity_minutes
        return timestamp.replace(minute=rounded_minutes, second=0, microsecond=0)


class GeospatialAnalyzer:
    """Analyzes geospatial patterns in sentiment data"""
    
    def __init__(self, grid_resolution: float = 0.1):  # degrees
        self.grid_resolution = grid_resolution
        self.location_sentiment: Dict[tuple, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self.clusterer = DBSCAN(eps=0.1, min_samples=5)
        
    def add_location_sentiment(self, lat: float, lon: float, sentiment: str) -> None:
        """Add sentiment data with location"""
        # Round to grid
        grid_lat = round(lat / self.grid_resolution) * self.grid_resolution
        grid_lon = round(lon / self.grid_resolution) * self.grid_resolution
        
        self.location_sentiment[(grid_lat, grid_lon)][sentiment] += 1
        
    def get_sentiment_heatmap_data(self) -> List[Dict[str, Any]]:
        """Get data for sentiment heatmap visualization"""
        heatmap_data = []
        
        for (lat, lon), sentiment_counts in self.location_sentiment.items():
            total_count = sum(sentiment_counts.values())
            if total_count == 0:
                continue
                
            # Calculate sentiment ratios
            positive_ratio = sentiment_counts.get('positive', 0) / total_count
            negative_ratio = sentiment_counts.get('negative', 0) / total_count
            neutral_ratio = sentiment_counts.get('neutral', 0) / total_count
            
            # Overall sentiment score (-1 to 1)
            sentiment_score = positive_ratio - negative_ratio
            
            heatmap_data.append({
                'lat': lat,
                'lon': lon,
                'total_count': total_count,
                'positive_ratio': positive_ratio,
                'negative_ratio': negative_ratio,
                'neutral_ratio': neutral_ratio,
                'sentiment_score': sentiment_score
            })
            
        return heatmap_data
    
    def find_sentiment_clusters(self) -> List[Dict[str, Any]]:
        """Find geographic clusters of similar sentiment"""
        if not self.location_sentiment:
            return []
            
        # Prepare data for clustering
        locations = []
        sentiment_features = []
        
        for (lat, lon), sentiment_counts in self.location_sentiment.items():
            total = sum(sentiment_counts.values())
            if total < 3:  # Skip sparse locations
                continue
                
            locations.append([lat, lon])
            
            # Normalize sentiment counts
            pos_ratio = sentiment_counts.get('positive', 0) / total
            neg_ratio = sentiment_counts.get('negative', 0) / total
            neu_ratio = sentiment_counts.get('neutral', 0) / total
            
            sentiment_features.append([pos_ratio, neg_ratio, neu_ratio])
        
        if len(locations) < 5:
            return []
            
        # Combine location and sentiment features
        features = np.hstack([
            StandardScaler().fit_transform(locations),
            StandardScaler().fit_transform(sentiment_features)
        ])
        
        # Perform clustering
        cluster_labels = self.clusterer.fit_predict(features)
        
        # Analyze clusters
        clusters = []
        for cluster_id in set(cluster_labels):
            if cluster_id == -1:  # Noise points
                continue
                
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            cluster_locations = [locations[i] for i in cluster_indices]
            cluster_sentiments = [sentiment_features[i] for i in cluster_indices]
            
            # Calculate cluster statistics
            avg_lat = np.mean([loc[0] for loc in cluster_locations])
            avg_lon = np.mean([loc[1] for loc in cluster_locations])
            avg_sentiment = np.mean(cluster_sentiments, axis=0)
            
            clusters.append({
                'cluster_id': cluster_id,
                'center_lat': avg_lat,
                'center_lon': avg_lon,
                'size': len(cluster_indices),
                'avg_positive_ratio': avg_sentiment[0],
                'avg_negative_ratio': avg_sentiment[1],
                'avg_neutral_ratio': avg_sentiment[2],
                'dominant_sentiment': ['positive', 'negative', 'neutral'][np.argmax(avg_sentiment)]
            })
            
        return clusters


class AlertSystem:
    """Real-time alert system for sentiment anomalies"""
    
    def __init__(self):
        self.alert_rules: List[Dict] = []
        self.alert_callbacks: List[Callable] = []
        self.alert_history: deque = deque(maxlen=1000)
        
    def add_alert_rule(self, rule: Dict[str, Any]) -> None:
        """Add alert rule
        
        Example rule:
        {
            'name': 'High Negative Sentiment',
            'condition': 'negative_ratio > 0.8',
            'threshold': 0.8,
            'severity': 'high',
            'cooldown_minutes': 15
        }
        """
        self.alert_rules.append({
            **rule,
            'last_triggered': None
        })
        
    def add_alert_callback(self, callback: Callable) -> None:
        """Add callback function for alert notifications"""
        self.alert_callbacks.append(callback)
        
    def check_alerts(self, metrics: AnalyticsMetrics) -> List[Dict[str, Any]]:
        """Check current metrics against alert rules"""
        triggered_alerts = []
        current_time = datetime.now()
        
        for rule in self.alert_rules:
            # Check cooldown
            if rule.get('last_triggered'):
                cooldown = timedelta(minutes=rule.get('cooldown_minutes', 15))
                if current_time - rule['last_triggered'] < cooldown:
                    continue
                    
            # Evaluate condition
            if self._evaluate_condition(rule, metrics):
                alert = {
                    'rule_name': rule['name'],
                    'severity': rule.get('severity', 'medium'),
                    'message': self._generate_alert_message(rule, metrics),
                    'timestamp': current_time,
                    'metrics_snapshot': metrics.to_dict()
                }
                
                triggered_alerts.append(alert)
                self.alert_history.append(alert)
                rule['last_triggered'] = current_time
                
                # Notify callbacks
                for callback in self.alert_callbacks:
                    try:
                        callback(alert)
                    except Exception as e:
                        logger.error(f"Error in alert callback: {e}")
                        
        return triggered_alerts
    
    def _evaluate_condition(self, rule: Dict, metrics: AnalyticsMetrics) -> bool:
        """Evaluate alert condition against current metrics"""
        condition = rule.get('condition', '')
        
        # Simple condition evaluation (could be enhanced with more complex expressions)
        if 'negative_ratio >' in condition:
            threshold = rule.get('threshold', 0.5)
            total_events = sum(metrics.sentiment_distribution.values())
            if total_events > 0:
                negative_ratio = metrics.sentiment_distribution.get('negative', 0) / total_events
                return negative_ratio > threshold
                
        elif 'positive_ratio <' in condition:
            threshold = rule.get('threshold', 0.3)
            total_events = sum(metrics.sentiment_distribution.values())
            if total_events > 0:
                positive_ratio = metrics.sentiment_distribution.get('positive', 0) / total_events
                return positive_ratio < threshold
                
        elif 'events_per_second >' in condition:
            threshold = rule.get('threshold', 100)
            return metrics.events_per_second > threshold
            
        elif 'average_confidence <' in condition:
            threshold = rule.get('threshold', 0.5)
            return metrics.average_confidence < threshold
            
        return False
    
    def _generate_alert_message(self, rule: Dict, metrics: AnalyticsMetrics) -> str:
        """Generate human-readable alert message"""
        return f"Alert: {rule['name']} triggered. Current metrics: {metrics.to_dict()}"


class RealTimeAnalyticsEngine:
    """Main real-time analytics engine"""
    
    def __init__(self, 
                 buffer_size: int = 10000,
                 metrics_update_interval: int = 5,
                 enable_geospatial: bool = True,
                 enable_timeseries: bool = True):
        
        # Core components
        self.stream_processor = StreamProcessor(buffer_size=buffer_size)
        self.metrics = AnalyticsMetrics()
        self.alert_system = AlertSystem()
        
        # Optional analyzers
        self.time_series_analyzer = TimeSeriesAnalyzer() if enable_timeseries else None
        self.geospatial_analyzer = GeospatialAnalyzer() if enable_geospatial else None
        
        # WebSocket connections
        self.websocket_clients: set = set()
        
        # Configuration
        self.metrics_update_interval = metrics_update_interval
        self._metrics_update_thread = None
        self._stop_metrics_update = threading.Event()
        
        # Register processing callbacks
        self.stream_processor.add_callback(self._process_event_batch)
        
        # Start components
        self._start_metrics_updates()
        
        logger.info("Real-time Analytics Engine initialized")
    
    def add_event(self, event: SentimentEvent) -> None:
        """Add sentiment event to processing pipeline"""
        self.stream_processor.push_event(event)
        
    def add_events_batch(self, events: List[SentimentEvent]) -> None:
        """Add batch of sentiment events"""
        for event in events:
            self.add_event(event)
    
    def _process_event_batch(self, events: List[SentimentEvent]) -> None:
        """Process batch of events for analytics"""
        for event in events:
            # Update basic metrics
            self.metrics.total_events += 1
            
            # Update sentiment distribution
            if event.sentiment not in self.metrics.sentiment_distribution:
                self.metrics.sentiment_distribution[event.sentiment] = 0
            self.metrics.sentiment_distribution[event.sentiment] += 1
            
            # Update source tracking
            if event.source not in self.metrics.top_sources:
                self.metrics.top_sources[event.source] = 0
            self.metrics.top_sources[event.source] += 1
            
            # Update confidence tracking
            total_confidence = (self.metrics.average_confidence * 
                              (self.metrics.total_events - 1) + event.confidence)
            self.metrics.average_confidence = total_confidence / self.metrics.total_events
            
            # Time series analysis
            if self.time_series_analyzer:
                self.time_series_analyzer.add_sentiment_point(event.sentiment, event.timestamp)
            
            # Geospatial analysis
            if self.geospatial_analyzer and event.location:
                self.geospatial_analyzer.add_location_sentiment(
                    event.location['lat'], event.location['lon'], event.sentiment
                )
                
                # Update geographic distribution
                region = f"{event.location['lat']:.1f},{event.location['lon']:.1f}"
                if region not in self.metrics.geographic_distribution:
                    self.metrics.geographic_distribution[region] = 0
                self.metrics.geographic_distribution[region] += 1
    
    def get_current_metrics(self) -> AnalyticsMetrics:
        """Get current analytics metrics"""
        # Update events per second
        if self.metrics.total_events > 0:
            time_elapsed = (datetime.now() - self.metrics.timestamp).total_seconds()
            if time_elapsed > 0:
                self.metrics.events_per_second = self.metrics.total_events / time_elapsed
        
        self.metrics.timestamp = datetime.now()
        return self.metrics
    
    def get_time_series_data(self, granularity_minutes: int = 5) -> Dict[str, List]:
        """Get time series sentiment data"""
        if not self.time_series_analyzer:
            return {}
        return self.time_series_analyzer.get_sentiment_trends(granularity_minutes)
    
    def get_geospatial_data(self) -> Dict[str, Any]:
        """Get geospatial sentiment analysis data"""
        if not self.geospatial_analyzer:
            return {}
            
        return {
            'heatmap_data': self.geospatial_analyzer.get_sentiment_heatmap_data(),
            'clusters': self.geospatial_analyzer.find_sentiment_clusters()
        }
    
    def get_anomalies(self) -> List[Dict[str, Any]]:
        """Get detected anomalies"""
        anomalies = []
        
        if self.time_series_analyzer:
            anomalies.extend(self.time_series_analyzer.detect_anomalies())
            
        return anomalies
    
    def add_alert_rule(self, rule: Dict[str, Any]) -> None:
        """Add custom alert rule"""
        self.alert_system.add_alert_rule(rule)
        
    def add_alert_callback(self, callback: Callable) -> None:
        """Add alert notification callback"""
        self.alert_system.add_alert_callback(callback)
    
    def _start_metrics_updates(self) -> None:
        """Start background metrics updates"""
        self._stop_metrics_update.clear()
        self._metrics_update_thread = threading.Thread(target=self._metrics_update_loop)
        self._metrics_update_thread.daemon = True
        self._metrics_update_thread.start()
        
        self.stream_processor.start_processing()
        
    def _metrics_update_loop(self) -> None:
        """Background loop for metrics updates and alerting"""
        while not self._stop_metrics_update.is_set():
            try:
                # Update metrics
                current_metrics = self.get_current_metrics()
                
                # Check alerts
                triggered_alerts = self.alert_system.check_alerts(current_metrics)
                
                # Broadcast to WebSocket clients
                self._broadcast_to_websockets({
                    'type': 'metrics_update',
                    'metrics': current_metrics.to_dict(),
                    'alerts': triggered_alerts,
                    'timestamp': datetime.now().isoformat()
                })
                
                time.sleep(self.metrics_update_interval)
                
            except Exception as e:
                logger.error(f"Error in metrics update loop: {e}")
                time.sleep(self.metrics_update_interval)
    
    def _broadcast_to_websockets(self, data: Dict[str, Any]) -> None:
        """Broadcast data to all connected WebSocket clients"""
        if not self.websocket_clients:
            return
            
        message = json.dumps(data, default=str)
        disconnected_clients = set()
        
        for client in self.websocket_clients:
            try:
                asyncio.create_task(client.send(message))
            except Exception:
                disconnected_clients.add(client)
        
        # Remove disconnected clients
        self.websocket_clients -= disconnected_clients
    
    async def websocket_handler(self, websocket, path):
        """Handle WebSocket connections for real-time updates"""
        self.websocket_clients.add(websocket)
        logger.info(f"WebSocket client connected: {websocket.remote_address}")
        
        try:
            # Send initial data
            initial_data = {
                'type': 'initial_data',
                'metrics': self.get_current_metrics().to_dict(),
                'time_series': self.get_time_series_data(),
                'geospatial': self.get_geospatial_data(),
                'anomalies': self.get_anomalies()
            }
            
            await websocket.send(json.dumps(initial_data, default=str))
            
            # Keep connection alive and handle incoming messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self._handle_websocket_message(websocket, data)
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({'error': 'Invalid JSON'}))
                    
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.websocket_clients.discard(websocket)
            logger.info(f"WebSocket client disconnected: {websocket.remote_address}")
    
    async def _handle_websocket_message(self, websocket, data: Dict[str, Any]) -> None:
        """Handle incoming WebSocket messages"""
        message_type = data.get('type')
        
        if message_type == 'get_metrics':
            response = {
                'type': 'metrics_response',
                'metrics': self.get_current_metrics().to_dict()
            }
            await websocket.send(json.dumps(response, default=str))
            
        elif message_type == 'get_time_series':
            granularity = data.get('granularity_minutes', 5)
            response = {
                'type': 'time_series_response',
                'data': self.get_time_series_data(granularity)
            }
            await websocket.send(json.dumps(response, default=str))
            
        elif message_type == 'add_alert_rule':
            rule = data.get('rule', {})
            self.add_alert_rule(rule)
            await websocket.send(json.dumps({'type': 'alert_rule_added'}))
    
    def generate_dashboard_html(self) -> str:
        """Generate HTML dashboard for real-time analytics"""
        if not PLOTLY_AVAILABLE:
            return "<html><body><h1>Plotly not available for dashboard generation</h1></body></html>"
        
        # Get current data
        metrics = self.get_current_metrics()
        time_series_data = self.get_time_series_data()
        geospatial_data = self.get_geospatial_data()
        
        # Create sentiment distribution pie chart
        sentiment_fig = px.pie(
            values=list(metrics.sentiment_distribution.values()),
            names=list(metrics.sentiment_distribution.keys()),
            title="Sentiment Distribution"
        )
        
        # Create time series chart
        time_series_fig = go.Figure()
        for sentiment, data_points in time_series_data.items():
            if data_points:
                timestamps = [point['timestamp'] for point in data_points]
                counts = [point['count'] for point in data_points]
                time_series_fig.add_trace(go.Scatter(
                    x=timestamps, y=counts, name=sentiment.title(), mode='lines+markers'
                ))
        
        time_series_fig.update_layout(title="Sentiment Trends Over Time")
        
        # Generate HTML
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Real-Time Sentiment Analytics</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .metric-box {{ 
                    display: inline-block; 
                    background: #f0f0f0; 
                    padding: 15px; 
                    margin: 10px; 
                    border-radius: 5px; 
                }}
                .chart-container {{ margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>Real-Time Sentiment Analytics Dashboard</h1>
            
            <div class="metrics-section">
                <div class="metric-box">
                    <h3>Total Events</h3>
                    <p>{metrics.total_events:,}</p>
                </div>
                <div class="metric-box">
                    <h3>Events/Second</h3>
                    <p>{metrics.events_per_second:.2f}</p>
                </div>
                <div class="metric-box">
                    <h3>Average Confidence</h3>
                    <p>{metrics.average_confidence:.2%}</p>
                </div>
            </div>
            
            <div class="chart-container">
                <div id="sentiment-chart">{sentiment_fig.to_html(include_plotlyjs=False)}</div>
            </div>
            
            <div class="chart-container">
                <div id="timeseries-chart">{time_series_fig.to_html(include_plotlyjs=False)}</div>
            </div>
            
            <script>
                // Auto-refresh every 30 seconds
                setTimeout(() => location.reload(), 30000);
            </script>
        </body>
        </html>
        """
        
        return html_template
    
    def shutdown(self) -> None:
        """Shutdown analytics engine"""
        self._stop_metrics_update.set()
        if self._metrics_update_thread:
            self._metrics_update_thread.join(timeout=5.0)
        
        self.stream_processor.stop_processing()
        logger.info("Real-time Analytics Engine shutdown")


# Factory function
def create_analytics_engine(**kwargs) -> RealTimeAnalyticsEngine:
    """Create real-time analytics engine with custom configuration"""
    return RealTimeAnalyticsEngine(**kwargs)


# Example usage
if __name__ == "__main__":
    # Create analytics engine
    engine = create_analytics_engine()
    
    # Add some alert rules
    engine.add_alert_rule({
        'name': 'High Negative Sentiment',
        'condition': 'negative_ratio > 0.8',
        'threshold': 0.8,
        'severity': 'high',
        'cooldown_minutes': 15
    })
    
    # Simulate some events
    import random
    sentiments = ['positive', 'negative', 'neutral']
    
    for i in range(100):
        event = SentimentEvent(
            text=f"Sample text {i}",
            sentiment=random.choice(sentiments),
            confidence=random.uniform(0.5, 1.0),
            source=random.choice(['twitter', 'facebook', 'instagram']),
            location={'lat': random.uniform(40.0, 41.0), 'lon': random.uniform(-74.0, -73.0)}
        )
        engine.add_event(event)
    
    # Get analytics
    print("Current Metrics:", engine.get_current_metrics().to_dict())
    print("Time Series:", engine.get_time_series_data())
    print("Geospatial:", engine.get_geospatial_data())
    print("Anomalies:", engine.get_anomalies())
    
    # Generate dashboard
    dashboard_html = engine.generate_dashboard_html()
    with open("/tmp/dashboard.html", "w") as f:
        f.write(dashboard_html)
    print("Dashboard saved to /tmp/dashboard.html")
    
    # Cleanup
    time.sleep(2)
    engine.shutdown()