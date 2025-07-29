# Observability and Monitoring Stack

This directory contains the observability and monitoring configuration for the Sentiment Analyzer Pro application.

## Architecture Overview

Our monitoring stack provides comprehensive observability across three pillars:
- **Metrics**: Performance and business metrics via Prometheus
- **Traces**: Distributed tracing via OpenTelemetry and Jaeger  
- **Logs**: Centralized logging via Loki and Promtail

## Quick Start

### 1. Start the Monitoring Stack
```bash
# Start all monitoring services
docker-compose -f docker-compose.monitoring.yml up -d

# View running services
docker-compose -f docker-compose.monitoring.yml ps
```

### 2. Access Dashboards
- **Grafana**: http://localhost:3000 (admin/admin123)
- **Prometheus**: http://localhost:9090
- **Jaeger**: http://localhost:16686
- **Application**: http://localhost:8080

### 3. Generate Some Data
```bash
# Make some API calls to generate metrics and traces
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this product!"}'

# Check health endpoint
curl http://localhost:8080/health
```

## Services Overview

### Core Application
- **sentiment-analyzer**: Main application with instrumentation
- **Port**: 8080
- **Health Check**: `/health`
- **Metrics**: `/metrics`

### Metrics Collection
- **Prometheus**: Time-series database for metrics
- **Port**: 9090
- **Config**: `prometheus.yml`
- **Alerts**: `alert_rules.yml`

### Visualization
- **Grafana**: Dashboards and alerting
- **Port**: 3000
- **Default Login**: admin/admin123
- **Data Sources**: Prometheus, Loki, Jaeger

### Distributed Tracing
- **OpenTelemetry Collector**: Trace processing and forwarding
- **Ports**: 4317 (gRPC), 4318 (HTTP)
- **Jaeger**: Trace storage and visualization
- **Port**: 16686

### Log Aggregation
- **Loki**: Log storage and querying
- **Port**: 3100
- **Promtail**: Log collection agent
- **Sources**: Application logs, system logs

### Caching Layer
- **Redis**: Session storage and caching
- **Port**: 6379
- **Password**: sentiment123

## Configuration Files

### prometheus.yml
Prometheus scraping configuration for all services:
- Application metrics on `:8080/metrics`
- OpenTelemetry Collector metrics
- System and container metrics

### alert_rules.yml
Alerting rules for common issues:
- High error rates (> 10%)
- High response times (> 1s)
- Service availability
- Resource utilization

### otel-collector.yml
OpenTelemetry Collector configuration:
- OTLP receivers for traces/metrics/logs
- Prometheus metrics export
- Jaeger trace export
- Log forwarding

## Monitoring Best Practices

### Application Instrumentation
```python
from opentelemetry import trace, metrics
from prometheus_client import Counter, Histogram

# Prometheus metrics
request_counter = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint'])
response_time = Histogram('http_request_duration_seconds', 'HTTP request duration')

# OpenTelemetry tracing
tracer = trace.get_tracer(__name__)

@response_time.time()
def process_request():
    request_counter.labels(method='POST', endpoint='/predict').inc()
    
    with tracer.start_as_current_span("model_prediction") as span:
        span.set_attribute("model.version", "1.0")
        # Your processing logic here
```

### Health Checks
```python
@app.route('/health')
def health():
    return {
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': app.config['VERSION'],
        'dependencies': {
            'database': check_database(),
            'cache': check_redis(),
            'model': check_model()
        }
    }
```

## Alerting Configuration

### Critical Alerts
- **Service Down**: Service unavailable for > 1 minute
- **High Error Rate**: Error rate > 10% for > 2 minutes
- **High Memory Usage**: Memory usage > 90% for > 5 minutes

### Warning Alerts
- **High Response Time**: 95th percentile > 1s for > 2 minutes
- **High CPU Usage**: CPU usage > 80% for > 5 minutes
- **Disk Space Low**: Disk usage > 90%

### Alert Destinations
Configure alerting destinations in `prometheus.yml`:
```yaml
alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

## Dashboard Templates

### Application Dashboard
Key metrics to monitor:
- Request rate and error rate
- Response time percentiles
- Active users and sessions
- Model prediction accuracy

### Infrastructure Dashboard
System metrics:
- CPU and memory utilization
- Disk I/O and network traffic
- Container resource usage
- Database performance

### Business Dashboard
Business-specific metrics:
- Sentiment prediction distribution
- Model confidence scores
- User interaction patterns
- Feature usage analytics

## Troubleshooting

### Common Issues

#### No Metrics Appearing
```bash
# Check if Prometheus can reach the application
curl http://localhost:8080/metrics

# Verify Prometheus configuration
docker-compose -f docker-compose.monitoring.yml logs prometheus
```

#### Missing Traces
```bash
# Check OpenTelemetry Collector logs
docker-compose -f docker-compose.monitoring.yml logs otel-collector

# Verify trace export configuration
curl http://localhost:4318/v1/traces -X POST -H "Content-Type: application/json" -d '{}'
```

#### Log Collection Issues
```bash
# Check Promtail status
docker-compose -f docker-compose.monitoring.yml logs promtail

# Verify log file permissions
ls -la ./logs/
```

### Performance Tuning

#### Prometheus Retention
Adjust retention in `docker-compose.monitoring.yml`:
```yaml
command:
  - '--storage.tsdb.retention.time=30d'
  - '--storage.tsdb.retention.size=10GB'
```

#### OpenTelemetry Sampling
Configure sampling in `otel-collector.yml`:
```yaml
processors:
  probabilistic_sampler:
    sampling_percentage: 10.0
```

## Security Considerations

### Access Control
- Use authentication for Grafana access
- Restrict Prometheus query access
- Secure inter-service communication

### Data Privacy
- Avoid logging sensitive information
- Sanitize metrics labels
- Implement data retention policies

### Network Security
- Use internal networks for service communication
- Enable TLS for external endpoints
- Regular security updates for base images

## Maintenance

### Regular Tasks
- **Weekly**: Review alert configurations
- **Monthly**: Check disk usage and retention
- **Quarterly**: Update dashboard templates
- **Annually**: Security audit and access review

### Backup Procedures
- Export Grafana dashboards and data sources
- Backup Prometheus configuration
- Document custom alert rules and thresholds

---

For more information, see:
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [OpenTelemetry Documentation](https://opentelemetry.io/docs/)
- [Jaeger Documentation](https://www.jaegertracing.io/docs/)