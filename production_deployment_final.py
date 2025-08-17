#!/usr/bin/env python3
"""Production Deployment System - Complete SDLC Implementation."""

import os
import sys
import json
import time
import subprocess
from datetime import datetime
from typing import Dict, Any, List
import logging
sys.path.insert(0, '/root/repo')

class ProductionDeploymentSystem:
    """Complete production deployment system for the sentiment analyzer."""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.deployment_config = self._load_deployment_config()
        self.start_time = time.time()
        
    def _setup_logging(self):
        """Setup deployment logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def _load_deployment_config(self):
        """Load deployment configuration."""
        return {
            "app_name": "sentiment-analyzer-pro",
            "version": "1.0.0",
            "environment": "production",
            "port": 5000,
            "workers": 4,
            "max_memory": "512M",
            "health_check_path": "/",
            "metrics_path": "/metrics",
            "prediction_path": "/predict"
        }
    
    def create_dockerfile(self):
        """Create optimized production Dockerfile."""
        self.logger.info("🐳 Creating production Dockerfile...")
        
        dockerfile_content = """# Production Dockerfile for Sentiment Analyzer Pro
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \\
    PYTHONDONTWRITEBYTECODE=1 \\
    PIP_NO_CACHE_DIR=1 \\
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    --no-install-recommends \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .
COPY pyproject.toml .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \\
    pip install --no-cache-dir flask gunicorn psutil requests

# Copy application code
COPY src/ ./src/
COPY data/ ./data/

# Create necessary directories and set permissions
RUN mkdir -p /app/logs /app/models && \\
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:5000/ || exit 1

# Default command
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--timeout", "30", "--keep-alive", "2", "--max-requests", "1000", "--max-requests-jitter", "100", "src.webapp:app"]
"""
        
        with open('/root/repo/Dockerfile.production', 'w') as f:
            f.write(dockerfile_content)
        
        self.logger.info("✅ Production Dockerfile created")
        return True
    
    def create_docker_compose(self):
        """Create production Docker Compose configuration."""
        self.logger.info("🐳 Creating Docker Compose configuration...")
        
        compose_content = """version: '3.8'

services:
  sentiment-analyzer:
    build:
      context: .
      dockerfile: Dockerfile.production
    container_name: sentiment-analyzer-pro
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
      - MODEL_PATH=/app/models
      - LOG_LEVEL=INFO
    volumes:
      - ./models:/app/models:ro
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    deploy:
      resources:
        limits:
          memory: 512M
          cpus: '1.0'
        reservations:
          memory: 256M
          cpus: '0.5'
    networks:
      - sentiment-network

  nginx:
    image: nginx:alpine
    container_name: sentiment-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
    depends_on:
      - sentiment-analyzer
    restart: unless-stopped
    networks:
      - sentiment-network

  prometheus:
    image: prom/prometheus:latest
    container_name: sentiment-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--web.enable-lifecycle'
    restart: unless-stopped
    networks:
      - sentiment-network

networks:
  sentiment-network:
    driver: bridge

volumes:
  prometheus-data:
"""
        
        with open('/root/repo/docker-compose.production.yml', 'w') as f:
            f.write(compose_content)
        
        self.logger.info("✅ Docker Compose configuration created")
        return True
    
    def create_nginx_config(self):
        """Create Nginx reverse proxy configuration."""
        self.logger.info("🌐 Creating Nginx configuration...")
        
        os.makedirs('/root/repo/nginx', exist_ok=True)
        
        nginx_content = """events {
    worker_connections 1024;
}

http {
    include       /etc/nginx/mime.types;
    default_type  application/octet-stream;
    
    # Logging
    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                    '$status $body_bytes_sent "$http_referer" '
                    '"$http_user_agent" "$http_x_forwarded_for"';
    
    access_log /var/log/nginx/access.log main;
    error_log /var/log/nginx/error.log;
    
    # Basic settings
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;
    
    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css text/xml text/javascript application/json application/javascript application/xml+rss;
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    
    upstream sentiment_backend {
        server sentiment-analyzer:5000;
        keepalive 32;
    }
    
    server {
        listen 80;
        server_name _;
        
        # Security headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";
        add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload";
        
        # Health check endpoint
        location /health {
            access_log off;
            return 200 "healthy\\n";
            add_header Content-Type text/plain;
        }
        
        # Rate limiting for API endpoints
        location /predict {
            limit_req zone=api burst=20 nodelay;
            
            proxy_pass http://sentiment_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Timeouts
            proxy_connect_timeout 5s;
            proxy_send_timeout 10s;
            proxy_read_timeout 30s;
        }
        
        # All other requests
        location / {
            proxy_pass http://sentiment_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Timeouts
            proxy_connect_timeout 5s;
            proxy_send_timeout 10s;
            proxy_read_timeout 30s;
        }
    }
}
"""
        
        with open('/root/repo/nginx/nginx.conf', 'w') as f:
            f.write(nginx_content)
        
        self.logger.info("✅ Nginx configuration created")
        return True
    
    def create_kubernetes_manifests(self):
        """Create Kubernetes deployment manifests."""
        self.logger.info("☸️ Creating Kubernetes manifests...")
        
        os.makedirs('/root/repo/k8s', exist_ok=True)
        
        # Deployment manifest
        deployment_yaml = """apiVersion: apps/v1
kind: Deployment
metadata:
  name: sentiment-analyzer
  labels:
    app: sentiment-analyzer
spec:
  replicas: 3
  selector:
    matchLabels:
      app: sentiment-analyzer
  template:
    metadata:
      labels:
        app: sentiment-analyzer
    spec:
      containers:
      - name: sentiment-analyzer
        image: sentiment-analyzer-pro:latest
        ports:
        - containerPort: 5000
        env:
        - name: FLASK_ENV
          value: "production"
        - name: LOG_LEVEL
          value: "INFO"
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /
            port: 5000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /
            port: 5000
          initialDelaySeconds: 5
          periodSeconds: 5
      imagePullPolicy: IfNotPresent
---
apiVersion: v1
kind: Service
metadata:
  name: sentiment-analyzer-service
spec:
  selector:
    app: sentiment-analyzer
  ports:
    - protocol: TCP
      port: 80
      targetPort: 5000
  type: LoadBalancer
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: sentiment-analyzer-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: sentiment-analyzer
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
"""
        
        with open('/root/repo/k8s/deployment.yaml', 'w') as f:
            f.write(deployment_yaml)
        
        self.logger.info("✅ Kubernetes manifests created")
        return True
    
    def create_monitoring_config(self):
        """Create monitoring configuration."""
        self.logger.info("📊 Creating monitoring configuration...")
        
        os.makedirs('/root/repo/monitoring', exist_ok=True)
        
        # Prometheus configuration
        prometheus_config = """global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

scrape_configs:
  - job_name: 'sentiment-analyzer'
    static_configs:
      - targets: ['sentiment-analyzer:5000']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'nginx'
    static_configs:
      - targets: ['nginx:80']
    metrics_path: '/metrics'
    scrape_interval: 30s

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
"""
        
        with open('/root/repo/monitoring/prometheus.yml', 'w') as f:
            f.write(prometheus_config)
        
        # Alert rules
        alert_rules = """groups:
- name: sentiment_analyzer_alerts
  rules:
  - alert: HighErrorRate
    expr: rate(flask_http_request_exceptions_total[5m]) > 0.1
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "High error rate detected"
      description: "Error rate is {{ $value }} errors per second"

  - alert: HighMemoryUsage
    expr: container_memory_usage_bytes / container_spec_memory_limit_bytes > 0.9
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "High memory usage"
      description: "Memory usage is above 90%"

  - alert: HighCPUUsage
    expr: rate(container_cpu_usage_seconds_total[5m]) > 0.8
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High CPU usage"
      description: "CPU usage is above 80%"
"""
        
        with open('/root/repo/monitoring/alert_rules.yml', 'w') as f:
            f.write(alert_rules)
        
        self.logger.info("✅ Monitoring configuration created")
        return True
    
    def create_deployment_scripts(self):
        """Create deployment automation scripts."""
        self.logger.info("🚀 Creating deployment scripts...")
        
        os.makedirs('/root/repo/scripts', exist_ok=True)
        
        # Build script
        build_script = """#!/bin/bash
set -e

echo "🔨 Building Sentiment Analyzer Pro for Production"

# Build Docker image
echo "Building Docker image..."
docker build -f Dockerfile.production -t sentiment-analyzer-pro:latest .

# Tag with version
VERSION=$(date +%Y%m%d-%H%M%S)
docker tag sentiment-analyzer-pro:latest sentiment-analyzer-pro:$VERSION

echo "✅ Build complete: sentiment-analyzer-pro:latest, sentiment-analyzer-pro:$VERSION"
"""
        
        with open('/root/repo/scripts/build.sh', 'w') as f:
            f.write(build_script)
        os.chmod('/root/repo/scripts/build.sh', 0o755)
        
        # Deploy script
        deploy_script = """#!/bin/bash
set -e

echo "🚀 Deploying Sentiment Analyzer Pro"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running"
    exit 1
fi

# Stop existing containers
echo "Stopping existing containers..."
docker-compose -f docker-compose.production.yml down

# Pull latest images
echo "Pulling latest images..."
docker-compose -f docker-compose.production.yml pull

# Start services
echo "Starting services..."
docker-compose -f docker-compose.production.yml up -d

# Wait for services to be ready
echo "Waiting for services to be ready..."
sleep 30

# Health check
echo "Performing health check..."
curl -f http://localhost:5000/ || {
    echo "❌ Health check failed"
    docker-compose -f docker-compose.production.yml logs sentiment-analyzer
    exit 1
}

echo "✅ Deployment successful"
echo "🌐 Application is running at http://localhost:5000"
echo "📊 Prometheus is running at http://localhost:9090"
"""
        
        with open('/root/repo/scripts/deploy.sh', 'w') as f:
            f.write(deploy_script)
        os.chmod('/root/repo/scripts/deploy.sh', 0o755)
        
        # Kubernetes deploy script
        k8s_deploy_script = """#!/bin/bash
set -e

echo "☸️ Deploying to Kubernetes"

# Apply manifests
kubectl apply -f k8s/deployment.yaml

# Wait for rollout
kubectl rollout status deployment/sentiment-analyzer

# Get service info
kubectl get service sentiment-analyzer-service

echo "✅ Kubernetes deployment complete"
"""
        
        with open('/root/repo/scripts/deploy-k8s.sh', 'w') as f:
            f.write(k8s_deploy_script)
        os.chmod('/root/repo/scripts/deploy-k8s.sh', 0o755)
        
        self.logger.info("✅ Deployment scripts created")
        return True
    
    def create_production_readme(self):
        """Create production deployment README."""
        self.logger.info("📚 Creating production README...")
        
        readme_content = """# Sentiment Analyzer Pro - Production Deployment

This directory contains production deployment configurations for the Sentiment Analyzer Pro application.

## Quick Start

### Docker Compose (Recommended)

1. **Build the application:**
   ```bash
   ./scripts/build.sh
   ```

2. **Deploy to production:**
   ```bash
   ./scripts/deploy.sh
   ```

3. **Access the application:**
   - API: http://localhost:5000
   - Metrics: http://localhost:5000/metrics
   - Prometheus: http://localhost:9090

### Kubernetes Deployment

1. **Build and push image:**
   ```bash
   ./scripts/build.sh
   docker push sentiment-analyzer-pro:latest
   ```

2. **Deploy to Kubernetes:**
   ```bash
   ./scripts/deploy-k8s.sh
   ```

## Architecture

```
Internet -> Nginx (Load Balancer) -> Sentiment Analyzer App -> Prometheus (Monitoring)
```

## Configuration Files

- `Dockerfile.production` - Optimized production Docker image
- `docker-compose.production.yml` - Multi-service deployment
- `nginx/nginx.conf` - Reverse proxy and load balancing
- `k8s/deployment.yaml` - Kubernetes manifests
- `monitoring/prometheus.yml` - Monitoring configuration

## Features

- ✅ **High Performance**: Gunicorn WSGI server with multiple workers
- ✅ **Load Balancing**: Nginx reverse proxy with rate limiting
- ✅ **Monitoring**: Prometheus metrics and alerting
- ✅ **Auto-scaling**: Kubernetes HPA for dynamic scaling
- ✅ **Health Checks**: Comprehensive health monitoring
- ✅ **Security**: Non-root containers, security headers
- ✅ **Resilience**: Restart policies and resource limits

## API Endpoints

- `GET /` - Health check
- `POST /predict` - Sentiment prediction
- `GET /metrics` - Prometheus metrics

## Monitoring

Prometheus metrics available at `/metrics`:
- Request rate and response times
- Error rates and status codes
- System resource usage
- Application-specific metrics

## Security

- Rate limiting (10 requests/second burst 20)
- Security headers (HSTS, XSS protection, etc.)
- Non-root container execution
- Resource limits and quotas

## Scaling

### Horizontal Scaling
- Docker Compose: Increase replicas in compose file
- Kubernetes: Automatic scaling based on CPU/memory usage

### Vertical Scaling
- Adjust resource limits in deployment configurations
- Increase worker count in Gunicorn configuration

## Troubleshooting

### Check application logs:
```bash
docker-compose -f docker-compose.production.yml logs sentiment-analyzer
```

### Check system resources:
```bash
docker stats
```

### Test API directly:
```bash
curl -X POST http://localhost:5000/predict \\
  -H "Content-Type: application/json" \\
  -d '{"text": "I love this product!"}'
```

## Maintenance

### Update application:
1. Update code
2. Run `./scripts/build.sh`
3. Run `./scripts/deploy.sh`

### Backup considerations:
- No persistent data to backup (stateless application)
- Monitor Prometheus data retention
- Backup configuration files

## Performance Tuning

### Application Level:
- Adjust Gunicorn workers: `--workers 4`
- Tune worker timeout: `--timeout 30`
- Configure keep-alive: `--keep-alive 2`

### Infrastructure Level:
- Increase container resources
- Add more replicas
- Tune Nginx worker connections

## Production Checklist

- [ ] SSL/TLS certificates configured
- [ ] Domain name configured
- [ ] Monitoring dashboards set up
- [ ] Alert notifications configured
- [ ] Log aggregation configured
- [ ] Backup procedures documented
- [ ] Disaster recovery plan created
- [ ] Performance baseline established
- [ ] Security scan completed
- [ ] Load testing performed

## Support

For issues and questions:
1. Check application logs
2. Review monitoring dashboards
3. Verify infrastructure health
4. Contact development team

---

**Production Ready**: This deployment configuration is optimized for production use with high availability, monitoring, and security best practices.
"""
        
        with open('/root/repo/PRODUCTION_DEPLOYMENT.md', 'w') as f:
            f.write(readme_content)
        
        self.logger.info("✅ Production README created")
        return True
    
    def run_deployment_preparation(self):
        """Run complete production deployment preparation."""
        self.logger.info("🚀 Starting Production Deployment Preparation")
        
        deployment_tasks = {
            'dockerfile': self.create_dockerfile(),
            'docker_compose': self.create_docker_compose(),
            'nginx_config': self.create_nginx_config(),
            'kubernetes_manifests': self.create_kubernetes_manifests(),
            'monitoring_config': self.create_monitoring_config(),
            'deployment_scripts': self.create_deployment_scripts(),
            'production_readme': self.create_production_readme()
        }
        
        success_count = sum(deployment_tasks.values())
        total_tasks = len(deployment_tasks)
        
        # Create deployment summary
        deployment_summary = {
            'timestamp': datetime.now().isoformat(),
            'deployment_preparation_status': 'SUCCESS' if success_count == total_tasks else 'PARTIAL',
            'tasks_completed': success_count,
            'total_tasks': total_tasks,
            'execution_time_seconds': round(time.time() - self.start_time, 2),
            'deployment_config': self.deployment_config,
            'task_results': deployment_tasks,
            'next_steps': [
                'Run ./scripts/build.sh to build production image',
                'Run ./scripts/deploy.sh to deploy with Docker Compose',
                'Configure SSL certificates for HTTPS',
                'Set up monitoring dashboards',
                'Configure alert notifications',
                'Perform load testing',
                'Set up log aggregation'
            ]
        }
        
        # Save deployment summary
        with open('/root/repo/deployment_summary.json', 'w') as f:
            json.dump(deployment_summary, f, indent=2)
        
        self.logger.info(f"📊 Deployment Preparation: {success_count}/{total_tasks} tasks completed")
        
        if success_count == total_tasks:
            self.logger.info("🎉 PRODUCTION DEPLOYMENT READY!")
            self.logger.info("📁 All configuration files created successfully")
            self.logger.info("🚀 Run './scripts/deploy.sh' to deploy to production")
        else:
            self.logger.warning("⚠️ Some deployment tasks failed. Check logs for details.")
        
        return success_count == total_tasks, deployment_summary

def main():
    """Run production deployment preparation."""
    deployment_system = ProductionDeploymentSystem()
    success, summary = deployment_system.run_deployment_preparation()
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)