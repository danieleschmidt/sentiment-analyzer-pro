"""Simple monitoring dashboard for Sentiment Analyzer Pro."""

import json
import time
from typing import Dict, Any, List
from flask import Flask, render_template_string, jsonify, request

from .metrics import metrics_collector


# HTML template for the dashboard
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sentiment Analyzer Pro - Monitoring Dashboard</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }
        .container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            max-width: 1200px;
            margin: 0 auto;
        }
        .card {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s;
        }
        .card:hover {
            transform: translateY(-2px);
        }
        .card h3 {
            margin-top: 0;
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }
        .metric {
            display: flex;
            justify-content: space-between;
            margin: 10px 0;
            padding: 8px;
            background: #f8f9fa;
            border-radius: 5px;
        }
        .metric-value {
            font-weight: bold;
            color: #667eea;
        }
        .status-good { color: #28a745; }
        .status-warning { color: #ffc107; }
        .status-error { color: #dc3545; }
        .refresh-btn {
            background: #667eea;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            margin: 10px;
        }
        .refresh-btn:hover {
            background: #5a6fd8;
        }
        .timestamp {
            font-size: 0.9em;
            color: #666;
            text-align: center;
            margin-top: 20px;
        }
        .chart-container {
            height: 200px;
            margin: 15px 0;
        }
        .recent-logs {
            max-height: 300px;
            overflow-y: auto;
            font-family: monospace;
            font-size: 0.8em;
        }
        .log-entry {
            padding: 5px;
            margin: 2px 0;
            border-left: 3px solid #667eea;
            background: #f8f9fa;
        }
        .log-error { border-left-color: #dc3545; }
        .log-warning { border-left-color: #ffc107; }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <div class="header">
        <h1>Sentiment Analyzer Pro</h1>
        <p>Monitoring Dashboard</p>
        <button class="refresh-btn" onclick="refreshData()">🔄 Refresh</button>
        <span id="auto-refresh">
            <label>
                <input type="checkbox" id="auto-refresh-toggle" onchange="toggleAutoRefresh()">
                Auto-refresh (30s)
            </label>
        </span>
    </div>

    <div class="container">
        <!-- System Status -->
        <div class="card">
            <h3>📊 System Status</h3>
            <div class="metric">
                <span>Prometheus Status:</span>
                <span class="metric-value" id="prometheus-status">Loading...</span>
            </div>
            <div class="metric">
                <span>Uptime:</span>
                <span class="metric-value" id="uptime">Loading...</span>
            </div>
            <div class="metric">
                <span>Last Updated:</span>
                <span class="metric-value" id="last-updated">Loading...</span>
            </div>
        </div>

        <!-- Request Metrics -->
        <div class="card">
            <h3>🌐 Request Metrics</h3>
            <div class="metric">
                <span>Total Requests:</span>
                <span class="metric-value" id="total-requests">Loading...</span>
            </div>
            <div class="metric">
                <span>Average Response Time:</span>
                <span class="metric-value" id="avg-response-time">Loading...</span>
            </div>
            <div class="metric">
                <span>Requests/minute:</span>
                <span class="metric-value" id="requests-per-minute">Loading...</span>
            </div>
            <div class="chart-container">
                <canvas id="requestChart"></canvas>
            </div>
        </div>

        <!-- Prediction Metrics -->
        <div class="card">
            <h3>🤖 Prediction Metrics</h3>
            <div class="metric">
                <span>Total Predictions:</span>
                <span class="metric-value" id="total-predictions">Loading...</span>
            </div>
            <div class="metric">
                <span>Average Prediction Time:</span>
                <span class="metric-value" id="avg-prediction-time">Loading...</span>
            </div>
            <div class="metric">
                <span>Average Text Length:</span>
                <span class="metric-value" id="avg-text-length">Loading...</span>
            </div>
            <div class="chart-container">
                <canvas id="predictionChart"></canvas>
            </div>
        </div>

        <!-- Recent Activity -->
        <div class="card">
            <h3>📝 Recent Activity</h3>
            <div class="recent-logs" id="recent-activity">
                <p>Loading recent activity...</p>
            </div>
        </div>

        <!-- Error Summary -->
        <div class="card">
            <h3>⚠️ Error Summary</h3>
            <div id="error-summary">
                <p>Loading error data...</p>
            </div>
        </div>

        <!-- Performance Overview -->
        <div class="card">
            <h3>⚡ Performance Overview</h3>
            <div class="chart-container">
                <canvas id="performanceChart"></canvas>
            </div>
        </div>
    </div>

    <div class="timestamp" id="dashboard-timestamp">
        Last updated: Loading...
    </div>

    <script>
        let autoRefreshInterval = null;
        let requestChart = null;
        let predictionChart = null;
        let performanceChart = null;
        
        // Initialize charts
        function initCharts() {
            const chartConfig = {
                type: 'line',
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    }
                }
            };
            
            requestChart = new Chart(document.getElementById('requestChart'), {
                ...chartConfig,
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Response Time (ms)',
                        data: [],
                        borderColor: 'rgb(102, 126, 234)',
                        backgroundColor: 'rgba(102, 126, 234, 0.1)',
                        tension: 0.1
                    }]
                }
            });
            
            predictionChart = new Chart(document.getElementById('predictionChart'), {
                ...chartConfig,
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Prediction Time (ms)',
                        data: [],
                        borderColor: 'rgb(118, 75, 162)',
                        backgroundColor: 'rgba(118, 75, 162, 0.1)',
                        tension: 0.1
                    }]
                }
            });
            
            performanceChart = new Chart(document.getElementById('performanceChart'), {
                type: 'doughnut',
                data: {
                    labels: ['Fast (<100ms)', 'Medium (100-500ms)', 'Slow (>500ms)'],
                    datasets: [{
                        data: [70, 25, 5],
                        backgroundColor: [
                            'rgba(40, 167, 69, 0.8)',
                            'rgba(255, 193, 7, 0.8)',
                            'rgba(220, 53, 69, 0.8)'
                        ]
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false
                }
            });
        }
        
        async function refreshData() {
            try {
                const response = await fetch('/metrics/json');
                const data = await response.json();
                
                updateSystemStatus(data);
                updateRequestMetrics(data);
                updatePredictionMetrics(data);
                updateRecentActivity(data);
                updateCharts(data);
                
                document.getElementById('dashboard-timestamp').textContent = 
                    `Last updated: ${new Date().toLocaleString()}`;
                    
            } catch (error) {
                console.error('Failed to refresh data:', error);
                document.getElementById('dashboard-timestamp').textContent = 
                    `Error updating dashboard: ${error.message}`;
            }
        }
        
        function updateSystemStatus(data) {
            document.getElementById('prometheus-status').textContent = 
                data.system.prometheus_available ? '✅ Available' : '❌ Unavailable';
            document.getElementById('prometheus-status').className = 
                'metric-value ' + (data.system.prometheus_available ? 'status-good' : 'status-warning');
            
            const uptime = Math.floor((Date.now() / 1000 - data.system.timestamp) / 60);
            document.getElementById('uptime').textContent = uptime + ' minutes';
            
            document.getElementById('last-updated').textContent = 
                new Date(data.system.timestamp * 1000).toLocaleTimeString();
        }
        
        function updateRequestMetrics(data) {
            document.getElementById('total-requests').textContent = data.requests.total;
            document.getElementById('avg-response-time').textContent = 
                (data.requests.avg_duration * 1000).toFixed(1) + ' ms';
            
            // Calculate requests per minute from recent data
            const recentRequests = data.requests.recent;
            const now = Date.now() / 1000;
            const oneMinuteAgo = now - 60;
            const recentCount = recentRequests.filter(r => r.timestamp > oneMinuteAgo).length;
            document.getElementById('requests-per-minute').textContent = recentCount;
        }
        
        function updatePredictionMetrics(data) {
            document.getElementById('total-predictions').textContent = data.predictions.total;
            document.getElementById('avg-prediction-time').textContent = 
                (data.predictions.avg_duration * 1000).toFixed(1) + ' ms';
            document.getElementById('avg-text-length').textContent = 
                Math.round(data.predictions.avg_text_length) + ' chars';
        }
        
        function updateRecentActivity(data) {
            const container = document.getElementById('recent-activity');
            const activities = [];
            
            // Add recent requests
            data.requests.recent.slice(-10).forEach(req => {
                activities.push({
                    timestamp: req.timestamp,
                    text: `${req.method} ${req.endpoint} - ${req.status} (${(req.duration * 1000).toFixed(0)}ms)`,
                    type: req.status >= 400 ? 'error' : req.status >= 300 ? 'warning' : 'info'
                });
            });
            
            // Add recent predictions
            data.predictions.recent.slice(-5).forEach(pred => {
                activities.push({
                    timestamp: pred.timestamp,
                    text: `Prediction: ${pred.model_type} model (${(pred.duration * 1000).toFixed(0)}ms, ${pred.text_length} chars)`,
                    type: 'info'
                });
            });
            
            // Sort by timestamp and display
            activities.sort((a, b) => b.timestamp - a.timestamp);
            container.innerHTML = activities.slice(0, 15).map(activity => 
                `<div class="log-entry log-${activity.type}">
                    <small>${new Date(activity.timestamp * 1000).toLocaleTimeString()}</small> - 
                    ${activity.text}
                </div>`
            ).join('');
        }
        
        function updateCharts(data) {
            // Update request chart
            const recentRequests = data.requests.recent.slice(-20);
            requestChart.data.labels = recentRequests.map(r => 
                new Date(r.timestamp * 1000).toLocaleTimeString()
            );
            requestChart.data.datasets[0].data = recentRequests.map(r => r.duration * 1000);
            requestChart.update();
            
            // Update prediction chart
            const recentPredictions = data.predictions.recent.slice(-20);
            predictionChart.data.labels = recentPredictions.map(p => 
                new Date(p.timestamp * 1000).toLocaleTimeString()
            );
            predictionChart.data.datasets[0].data = recentPredictions.map(p => p.duration * 1000);
            predictionChart.update();
        }
        
        function toggleAutoRefresh() {
            const checkbox = document.getElementById('auto-refresh-toggle');
            if (checkbox.checked) {
                autoRefreshInterval = setInterval(refreshData, 30000); // 30 seconds
            } else {
                if (autoRefreshInterval) {
                    clearInterval(autoRefreshInterval);
                    autoRefreshInterval = null;
                }
            }
        }
        
        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', function() {
            initCharts();
            refreshData();
        });
    </script>
</body>
</html>
"""


def create_dashboard_app() -> Flask:
    """Create a Flask app for the monitoring dashboard."""
    app = Flask(__name__)
    
    @app.route('/dashboard')
    def dashboard():
        """Serve the monitoring dashboard."""
        return render_template_string(DASHBOARD_HTML)
    
    @app.route('/dashboard/api/metrics')
    def dashboard_metrics():
        """API endpoint for dashboard metrics."""
        dashboard_data = metrics_collector.get_dashboard_data()
        
        # Add additional computed metrics for dashboard
        dashboard_data['computed'] = compute_dashboard_metrics(dashboard_data)
        
        return jsonify(dashboard_data)
    
    return app


def compute_dashboard_metrics(data: Dict[str, Any]) -> Dict[str, Any]:
    """Compute additional metrics for dashboard display."""
    computed = {}
    
    # Request rate analysis
    recent_requests = data['requests']['recent']
    if recent_requests:
        now = time.time()
        
        # Calculate requests in different time windows
        minute_ago = now - 60
        hour_ago = now - 3600
        
        requests_last_minute = len([r for r in recent_requests if r['timestamp'] > minute_ago])
        requests_last_hour = len([r for r in recent_requests if r['timestamp'] > hour_ago])
        
        computed['requests_per_minute'] = requests_last_minute
        computed['requests_per_hour'] = requests_last_hour
        
        # Response time percentiles
        durations = [r['duration'] for r in recent_requests[-100:]]  # Last 100 requests
        if durations:
            durations.sort()
            n = len(durations)
            computed['response_time_p50'] = durations[n // 2] if n > 0 else 0
            computed['response_time_p95'] = durations[int(n * 0.95)] if n > 0 else 0
            computed['response_time_p99'] = durations[int(n * 0.99)] if n > 0 else 0
        
        # Error rate
        error_requests = [r for r in recent_requests if r['status'] >= 400]
        computed['error_rate'] = len(error_requests) / len(recent_requests) if recent_requests else 0
    
    # Prediction analysis
    recent_predictions = data['predictions']['recent']
    if recent_predictions:
        # Text length distribution
        text_lengths = [p['text_length'] for p in recent_predictions]
        if text_lengths:
            text_lengths.sort()
            n = len(text_lengths)
            computed['text_length_median'] = text_lengths[n // 2] if n > 0 else 0
            computed['text_length_p95'] = text_lengths[int(n * 0.95)] if n > 0 else 0
        
        # Prediction time distribution
        pred_durations = [p['duration'] for p in recent_predictions]
        if pred_durations:
            pred_durations.sort()
            n = len(pred_durations)
            computed['prediction_time_p50'] = pred_durations[n // 2] if n > 0 else 0
            computed['prediction_time_p95'] = pred_durations[int(n * 0.95)] if n > 0 else 0
    
    # System health indicators
    computed['health_score'] = calculate_health_score(data, computed)
    
    return computed


def calculate_health_score(data: Dict[str, Any], computed: Dict[str, Any]) -> float:
    """Calculate an overall system health score (0-100)."""
    score = 100.0
    
    # Deduct points for high error rate
    error_rate = computed.get('error_rate', 0)
    if error_rate > 0.05:  # More than 5% errors
        score -= min(error_rate * 200, 50)  # Max 50 point deduction
    
    # Deduct points for high response times
    p95_response = computed.get('response_time_p95', 0)
    if p95_response > 1.0:  # More than 1 second
        score -= min((p95_response - 1.0) * 20, 30)  # Max 30 point deduction
    
    # Deduct points if Prometheus is not available
    if not data['system'].get('prometheus_available', False):
        score -= 10
    
    # Boost score for high activity (good sign)
    requests_per_minute = computed.get('requests_per_minute', 0)
    if requests_per_minute > 10:
        score += min((requests_per_minute - 10) * 0.5, 5)  # Max 5 point bonus
    
    return max(0, min(100, score))


def add_dashboard_routes(app: Flask) -> None:
    """Add dashboard routes to an existing Flask app."""
    @app.route('/dashboard')
    def dashboard():
        """Serve the monitoring dashboard."""
        return render_template_string(DASHBOARD_HTML)
    
    @app.route('/dashboard/api/metrics')
    def dashboard_metrics():
        """API endpoint for dashboard metrics."""
        dashboard_data = metrics_collector.get_dashboard_data()
        dashboard_data['computed'] = compute_dashboard_metrics(dashboard_data)
        return jsonify(dashboard_data)


if __name__ == '__main__':
    # Standalone dashboard server
    dashboard_app = create_dashboard_app()
    
    # Add the metrics endpoint for compatibility
    @dashboard_app.route('/metrics/json')
    def metrics_json():
        """Metrics endpoint for dashboard."""
        dashboard_data = metrics_collector.get_dashboard_data()
        dashboard_data['computed'] = compute_dashboard_metrics(dashboard_data)
        return jsonify(dashboard_data)
    
    print("Starting monitoring dashboard on http://localhost:5001/dashboard")
    dashboard_app.run(host='0.0.0.0', port=5001, debug=True)