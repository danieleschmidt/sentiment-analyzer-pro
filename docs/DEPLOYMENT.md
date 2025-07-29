# Deployment Guide

This document provides comprehensive deployment strategies and infrastructure configurations for Sentiment Analyzer Pro.

## Deployment Options

### 1. Local Development Deployment

```bash
# Quick start for development
make setup
make dev

# Or using Docker
docker build -t sentiment-pro .
docker run -p 5000:5000 sentiment-pro
```

### 2. Production Docker Deployment

#### Multi-stage Dockerfile Optimization

```dockerfile
# Production-optimized Dockerfile
FROM python:3.10-slim as builder

WORKDIR /app
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.10-slim as runtime
WORKDIR /app

# Copy from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application
COPY src/ ./src/
COPY data/ ./data/

# Security: Run as non-root user
RUN useradd --create-home --shell /bin/bash sentiment
USER sentiment

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:5000/ || exit 1

EXPOSE 5000
CMD ["python", "-m", "src.webapp"]
```

#### Docker Compose for Production

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  sentiment-api:
    build: 
      context: .
      dockerfile: Dockerfile
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
      - MODEL_PATH=/app/models/production_model.joblib
    volumes:
      - ./models:/app/models:ro
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/"]
      interval: 30s
      timeout: 10s
      retries: 3
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.sentiment.rule=Host(`sentiment.yourdomain.com`)"

  reverse-proxy:
    image: traefik:v2.9
    command:
      - --api.dashboard=true
      - --providers.docker=true
      - --entrypoints.web.address=:80
      - --entrypoints.websecure.address=:443
      - --certificatesresolvers.letsencrypt.acme.tlschallenge=true
      - --certificatesresolvers.letsencrypt.acme.email=your-email@domain.com
      - --certificatesresolvers.letsencrypt.acme.storage=/acme.json
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:ro
      - ./acme.json:/acme.json
    restart: unless-stopped
```

### 3. Kubernetes Deployment

#### Basic Kubernetes Configuration

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
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
        - name: MODEL_PATH
          value: "/app/models/production_model.joblib"
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
```

#### Ingress Configuration

```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: sentiment-analyzer-ingress
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
spec:
  tls:
  - hosts:
    - sentiment.yourdomain.com
    secretName: sentiment-tls
  rules:
  - host: sentiment.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: sentiment-analyzer-service
            port:
              number: 80
```

### 4. Cloud Platform Deployments

#### AWS ECS with Fargate

```json
{
  "family": "sentiment-analyzer",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "256",
  "memory": "512",
  "executionRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "sentiment-analyzer",
      "image": "your-account.dkr.ecr.region.amazonaws.com/sentiment-analyzer:latest",
      "portMappings": [
        {
          "containerPort": 5000,
          "protocol": "tcp"
        }
      ],
      "essential": true,
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/sentiment-analyzer",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:5000/ || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3
      }
    }
  ]
}
```

#### Google Cloud Run

```yaml
# cloud-run.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: sentiment-analyzer
  annotations:
    run.googleapis.com/ingress: all
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/maxScale: "100"
        run.googleapis.com/cpu-throttling: "false"
    spec:
      containerConcurrency: 80
      containers:
      - image: gcr.io/PROJECT-ID/sentiment-analyzer
        ports:
        - containerPort: 5000
        env:
        - name: FLASK_ENV
          value: production
        resources:
          limits:
            cpu: 1000m
            memory: 512Mi
```

#### Azure Container Instances

```yaml
# azure-container-instance.yaml
apiVersion: 2019-12-01
location: eastus
name: sentiment-analyzer
properties:
  containers:
  - name: sentiment-analyzer
    properties:
      image: your-registry.azurecr.io/sentiment-analyzer:latest
      resources:
        requests:
          cpu: 1
          memory: 1
      ports:
      - port: 5000
        protocol: TCP
  osType: Linux
  restartPolicy: Always
  ipAddress:
    type: Public
    ports:
    - protocol: tcp
      port: '5000'
```

## Infrastructure as Code

### Terraform Configuration

```hcl
# terraform/main.tf
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# ECS Cluster
resource "aws_ecs_cluster" "sentiment_cluster" {
  name = "sentiment-analyzer"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

# Load Balancer
resource "aws_lb" "sentiment_lb" {
  name               = "sentiment-analyzer-lb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.lb_sg.id]
  subnets           = var.public_subnet_ids

  enable_deletion_protection = false
}

# Auto Scaling
resource "aws_appautoscaling_target" "sentiment_target" {
  max_capacity       = 10
  min_capacity       = 2
  resource_id        = "service/${aws_ecs_cluster.sentiment_cluster.name}/${aws_ecs_service.sentiment_service.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
}

resource "aws_appautoscaling_policy" "sentiment_up" {
  name               = "sentiment-scale-up"
  policy_type        = "StepScaling"
  resource_id        = aws_appautoscaling_target.sentiment_target.resource_id
  scalable_dimension = aws_appautoscaling_target.sentiment_target.scalable_dimension
  service_namespace  = aws_appautoscaling_target.sentiment_target.service_namespace

  step_scaling_policy_configuration {
    adjustment_type         = "ChangeInCapacity"
    cooldown               = 60
    metric_aggregation_type = "Maximum"

    step_adjustment {
      metric_interval_lower_bound = 0
      scaling_adjustment          = 1
    }
  }
}
```

### Ansible Playbook

```yaml
# ansible/deploy.yml
---
- name: Deploy Sentiment Analyzer
  hosts: production
  become: yes
  vars:
    app_name: sentiment-analyzer-pro
    app_port: 5000
    
  tasks:
    - name: Update system packages
      apt:
        update_cache: yes
        upgrade: dist

    - name: Install Docker
      apt:
        name: docker.io
        state: present

    - name: Install Docker Compose
      pip:
        name: docker-compose
        state: present

    - name: Create application directory
      file:
        path: /opt/{{ app_name }}
        state: directory
        mode: '0755'

    - name: Copy application files
      copy:
        src: "{{ item }}"
        dest: /opt/{{ app_name }}/
      with_items:
        - docker-compose.prod.yml
        - Dockerfile
        - src/
        - data/

    - name: Build and start containers
      docker_compose:
        project_src: /opt/{{ app_name }}
        file: docker-compose.prod.yml
        state: present

    - name: Configure firewall
      ufw:
        rule: allow
        port: "{{ app_port }}"
        proto: tcp
```

## Monitoring and Observability

### Prometheus Configuration

```yaml
# prometheus/prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'sentiment-analyzer'
    static_configs:
      - targets: ['sentiment-analyzer:5000']
    metrics_path: '/metrics'
    scrape_interval: 30s
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "Sentiment Analyzer Metrics",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "Requests/sec"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, http_request_duration_seconds_bucket)",
            "legendFormat": "95th percentile"
          }
        ]
      }
    ]
  }
}
```

## Security Configurations

### SSL/TLS Configuration

```nginx
# nginx/ssl.conf
server {
    listen 443 ssl http2;
    server_name sentiment.yourdomain.com;

    ssl_certificate /etc/ssl/certs/sentiment.crt;
    ssl_certificate_key /etc/ssl/private/sentiment.key;
    
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA384:ECDHE-RSA-AES128-GCM-SHA256;
    ssl_prefer_server_ciphers off;
    
    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Content-Type-Options nosniff;
    add_header X-Frame-Options DENY;
    add_header X-XSS-Protection "1; mode=block";
    
    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## Backup and Disaster Recovery

### Database Backup Strategy

```bash
#!/bin/bash
# scripts/backup.sh

# Model backup
tar -czf "models-backup-$(date +%Y%m%d).tar.gz" models/

# Upload to S3
aws s3 cp "models-backup-$(date +%Y%m%d).tar.gz" s3://your-backup-bucket/

# Clean old backups (keep last 30 days)
find . -name "models-backup-*.tar.gz" -mtime +30 -delete
```

### Disaster Recovery Plan

1. **RTO (Recovery Time Objective)**: 15 minutes
2. **RPO (Recovery Point Objective)**: 1 hour
3. **Backup Strategy**: Daily automated backups
4. **Failover Strategy**: Multi-region deployment with load balancer

## Scaling Strategies

### Horizontal Scaling

```yaml
# Auto-scaling configuration
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: sentiment-analyzer-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: sentiment-analyzer
  minReplicas: 3
  maxReplicas: 50
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
```

### Vertical Scaling

```yaml
# Vertical Pod Autoscaler
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: sentiment-analyzer-vpa
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: sentiment-analyzer
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: sentiment-analyzer
      maxAllowed:
        cpu: 2
        memory: 2Gi
      minAllowed:
        cpu: 100m
        memory: 128Mi
```

## Deployment Checklist

### Pre-deployment

- [ ] Run full test suite (`make test`)
- [ ] Security scan passed (`make security`)
- [ ] Performance benchmarks acceptable
- [ ] Documentation updated
- [ ] Dependencies updated and secure

### Deployment

- [ ] Environment variables configured
- [ ] SSL certificates valid
- [ ] Health checks configured
- [ ] Monitoring enabled
- [ ] Logging configured
- [ ] Backup strategy implemented

### Post-deployment

- [ ] Health checks passing
- [ ] Monitoring dashboards configured
- [ ] Load testing completed
- [ ] Rollback plan verified
- [ ] Documentation updated with deployment details

This deployment guide provides multiple options from simple local deployment to enterprise-grade cloud deployments with full observability and disaster recovery capabilities.