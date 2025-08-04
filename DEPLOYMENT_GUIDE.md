# Photonic-MLIR Bridge Deployment Guide

## 🚀 Production Deployment Configuration

This guide provides comprehensive instructions for deploying the Photonic-MLIR Bridge in production environments.

## System Requirements

### Minimum Requirements
- **CPU**: 2 cores
- **RAM**: 4GB
- **Storage**: 10GB available space
- **OS**: Linux (Ubuntu 20.04+), macOS 10.15+, Windows 10+
- **Python**: 3.9 or higher

### Recommended Requirements
- **CPU**: 8+ cores
- **RAM**: 16GB+
- **Storage**: 50GB+ SSD
- **OS**: Ubuntu 22.04 LTS
- **Python**: 3.11+
- **Network**: High-bandwidth for distributed synthesis

## Installation

### Production Installation

```bash
# Clone repository
git clone https://github.com/your-org/photonic-mlir-synth-bridge.git
cd photonic-mlir-synth-bridge

# Create production environment
python3 -m venv venv-prod
source venv-prod/bin/activate

# Install production dependencies
pip install -e .

# Verify installation
python3 -c "from src.photonic_init import get_photonic_status; print(get_photonic_status())"
```

### Docker Deployment

#### Single Container

```dockerfile
# Dockerfile.photonic
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY examples/ ./examples/
COPY tests/ ./tests/

# Set environment variables
ENV PYTHONPATH=/app
ENV PHOTONIC_LOG_LEVEL=INFO
ENV PHOTONIC_CACHE_SIZE=1000
ENV PHOTONIC_MAX_WORKERS=4

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "from src.photonic_monitoring import get_monitor; print('OK')" || exit 1

# Expose port for web interface (if enabled)
EXPOSE 8080

# Run application
CMD ["python3", "-m", "src.photonic_cli", "serve", "--host", "0.0.0.0", "--port", "8080"]
```

#### Docker Compose

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  photonic-bridge:
    build:
      context: .
      dockerfile: Dockerfile.photonic
    container_name: photonic-bridge
    restart: unless-stopped
    ports:
      - "8080:8080"
    environment:
      - PHOTONIC_ENV=production
      - PHOTONIC_LOG_LEVEL=INFO
      - PHOTONIC_CACHE_SIZE=5000
      - PHOTONIC_MAX_WORKERS=8
      - PHOTONIC_ENABLE_METRICS=true
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./output:/app/output
    networks:
      - photonic-network
    healthcheck:
      test: ["CMD", "python3", "-c", "from src.photonic_monitoring import get_monitor; print('OK')"]
      interval: 30s
      timeout: 10s
      retries: 3

  monitoring:
    image: prom/prometheus:latest
    container_name: photonic-prometheus
    restart: unless-stopped
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    networks:
      - photonic-network

  grafana:
    image: grafana/grafana:latest
    container_name: photonic-grafana
    restart: unless-stopped
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana:/etc/grafana/provisioning
    networks:
      - photonic-network

networks:
  photonic-network:
    driver: bridge

volumes:
  prometheus_data:
  grafana_data:
```

## Configuration

### Environment Variables

```bash
# Core configuration
export PHOTONIC_ENV=production
export PHOTONIC_LOG_LEVEL=INFO
export PHOTONIC_DEBUG=false

# Performance tuning
export PHOTONIC_CACHE_SIZE=5000
export PHOTONIC_MAX_WORKERS=8
export PHOTONIC_ENABLE_OPTIMIZATION=true

# Security settings
export PHOTONIC_RATE_LIMIT_WINDOW=60
export PHOTONIC_RATE_LIMIT_MAX_REQUESTS=1000
export PHOTONIC_SECURITY_LEVEL=strict

# Storage and output
export PHOTONIC_DATA_DIR=/app/data
export PHOTONIC_OUTPUT_DIR=/app/output
export PHOTONIC_LOG_DIR=/app/logs

# Monitoring and metrics
export PHOTONIC_ENABLE_METRICS=true
export PHOTONIC_METRICS_PORT=8081
export PHOTONIC_HEALTH_CHECK_PORT=8082
```

### Configuration File

```json
{
  "environment": "production",
  "logging": {
    "level": "INFO",
    "format": "json",
    "file": "/app/logs/photonic.log",
    "max_size_mb": 100,
    "backup_count": 5
  },
  "performance": {
    "cache_size": 5000,
    "max_workers": 8,
    "optimization_level": 2,
    "enable_concurrent_processing": true
  },
  "security": {
    "validation_level": "strict",
    "rate_limiting": {
      "enabled": true,
      "window_seconds": 60,
      "max_requests": 1000
    },
    "input_sanitization": true,
    "security_headers": true
  },
  "storage": {
    "data_directory": "/app/data",
    "output_directory": "/app/output",
    "temp_directory": "/tmp/photonic",
    "max_file_size_mb": 500
  },
  "monitoring": {
    "enabled": true,
    "metrics_endpoint": "/metrics",
    "health_endpoint": "/health",
    "prometheus_integration": true
  }
}
```

## Monitoring and Observability

### Health Checks

The system provides comprehensive health checks:

```bash
# Check system health
curl http://localhost:8080/health

# Check detailed metrics
curl http://localhost:8080/metrics

# Prometheus metrics
curl http://localhost:8081/metrics
```

### Logging Configuration

```python
# logging_config.py
import logging.config

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'detailed': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        },
        'json': {
            'format': '{"timestamp": "%(asctime)s", "logger": "%(name)s", "level": "%(levelname)s", "message": "%(message)s"}'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'detailed',
            'stream': 'ext://sys.stdout'
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'DEBUG',
            'formatter': 'json',
            'filename': '/app/logs/photonic.log',
            'maxBytes': 104857600,  # 100MB
            'backupCount': 5
        }
    },
    'loggers': {
        'src.photonic_mlir_bridge': {
            'level': 'INFO',
            'handlers': ['console', 'file'],
            'propagate': False
        },
        'src.photonic_monitoring': {
            'level': 'INFO',
            'handlers': ['console', 'file'],
            'propagate': False
        }
    },
    'root': {
        'level': 'INFO',
        'handlers': ['console', 'file']
    }
}

logging.config.dictConfig(LOGGING_CONFIG)
```

## Performance Optimization

### Caching Strategy

```python
# Enable advanced caching
CACHE_CONFIG = {
    "synthesis_cache": {
        "size": 5000,
        "ttl_seconds": 3600,
        "policy": "lru"
    },
    "validation_cache": {
        "size": 2000,  
        "ttl_seconds": 1800,
        "policy": "ttl"
    },
    "component_cache": {
        "size": 10000,
        "policy": "lfu"
    }
}
```

### Scaling Configuration

```yaml
# kubernetes-deployment.yml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: photonic-bridge
  labels:
    app: photonic-bridge
spec:
  replicas: 3
  selector:
    matchLabels:
      app: photonic-bridge
  template:
    metadata:
      labels:
        app: photonic-bridge
    spec:
      containers:
      - name: photonic-bridge
        image: photonic-bridge:latest
        ports:
        - containerPort: 8080
        env:
        - name: PHOTONIC_MAX_WORKERS
          value: "8"
        - name: PHOTONIC_CACHE_SIZE
          value: "5000"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "8Gi"
            cpu: "4000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: photonic-bridge-service
spec:
  selector:
    app: photonic-bridge
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8080
  type: LoadBalancer
```

## Security Hardening

### Security Checklist

- [ ] Enable input validation and sanitization
- [ ] Configure rate limiting
- [ ] Set up SSL/TLS encryption
- [ ] Enable security headers
- [ ] Configure firewall rules
- [ ] Set up intrusion detection
- [ ] Enable audit logging
- [ ] Regular security updates

### SSL/TLS Configuration

```nginx
# nginx.conf
server {
    listen 443 ssl http2;
    server_name photonic.example.com;

    ssl_certificate /etc/ssl/certs/photonic.crt;
    ssl_certificate_key /etc/ssl/private/photonic.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;

    # Security headers
    add_header Strict-Transport-Security "max-age=63072000" always;
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";

    location / {
        proxy_pass http://localhost:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## Backup and Recovery

### Database Backup

```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/app/backups"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup circuit data
tar -czf $BACKUP_DIR/circuits_$DATE.tar.gz /app/data/circuits/

# Backup configuration
cp /app/config/production.json $BACKUP_DIR/config_$DATE.json

# Backup logs (last 7 days)
find /app/logs -name "*.log" -mtime -7 -exec tar -czf $BACKUP_DIR/logs_$DATE.tar.gz {} \;

# Clean old backups (keep 30 days)
find $BACKUP_DIR -name "*.tar.gz" -mtime +30 -delete
find $BACKUP_DIR -name "*.json" -mtime +30 -delete

echo "Backup completed: $DATE"
```

### Disaster Recovery

```bash
#!/bin/bash
# restore.sh

BACKUP_FILE=$1
RESTORE_DIR="/app/restore"

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup_file>"
    exit 1
fi

# Create restore directory
mkdir -p $RESTORE_DIR

# Extract backup
tar -xzf $BACKUP_FILE -C $RESTORE_DIR

# Stop services
docker-compose down

# Restore data
cp -r $RESTORE_DIR/data/* /app/data/

# Start services
docker-compose up -d

echo "Restore completed from: $BACKUP_FILE"
```

## Troubleshooting

### Common Issues

1. **High Memory Usage**
   ```bash
   # Check memory usage
   docker stats photonic-bridge
   
   # Reduce cache size
   export PHOTONIC_CACHE_SIZE=1000
   ```

2. **Slow Performance**
   ```bash
   # Enable optimization
   export PHOTONIC_ENABLE_OPTIMIZATION=true
   
   # Increase worker count
   export PHOTONIC_MAX_WORKERS=16
   ```

3. **Connection Issues**
   ```bash
   # Check network connectivity
   curl -I http://localhost:8080/health
   
   # Check logs
   docker logs photonic-bridge
   ```

### Performance Monitoring

```bash
# Monitor synthesis performance
curl http://localhost:8080/metrics | grep synthesis

# Check cache hit rates
curl http://localhost:8080/metrics | grep cache

# Monitor system resources
htop
iostat -x 1
```

## Maintenance

### Regular Maintenance Tasks

1. **Weekly**
   - Review logs for errors
   - Check disk space usage
   - Verify backup integrity
   - Update security patches

2. **Monthly**
   - Performance optimization review
   - Cache statistics analysis
   - Security audit
   - Dependency updates

3. **Quarterly**
   - Full system backup
   - Disaster recovery testing
   - Performance benchmarking
   - Security penetration testing

### Automated Maintenance

```bash
#!/bin/bash
# maintenance.sh

# Log rotation
logrotate /etc/logrotate.d/photonic

# Cache cleanup
curl -X POST http://localhost:8080/admin/cache/clear

# System health check
python3 quality_gates.py

# Performance report
curl http://localhost:8080/admin/performance/report

echo "Maintenance completed: $(date)"
```

## Support and Contact

For production support and issues:

- **Documentation**: [https://docs.photonic-mlir.com](https://docs.photonic-mlir.com)
- **Issues**: [https://github.com/your-org/photonic-mlir-bridge/issues](https://github.com/your-org/photonic-mlir-bridge/issues)
- **Security**: security@photonic-mlir.com
- **Support**: support@photonic-mlir.com

---

**Production Deployment Checklist:**

- [ ] System requirements verified
- [ ] Environment variables configured
- [ ] SSL/TLS certificates installed
- [ ] Monitoring and alerting setup
- [ ] Backup strategy implemented
- [ ] Security hardening completed
- [ ] Performance optimization configured
- [ ] Health checks validated
- [ ] Documentation updated
- [ ] Team training completed