# Deployment Guide

This guide covers deployment options for the Sentiment Analyzer Pro application.

## Docker Deployment

### Quick Start with Docker

1. **Build the production image:**
   ```bash
   docker build -t sentiment-analyzer-pro .
   ```

2. **Run the application:**
   ```bash
   docker run -p 5000:8080 sentiment-analyzer-pro
   ```

3. **Access the application:**
   - API: http://localhost:5000
   - Health check: http://localhost:5000/

### Docker Compose Deployment

#### Production Setup
```bash
# Start the application
docker-compose up -d

# View logs
docker-compose logs -f sentiment-analyzer

# Stop the application
docker-compose down
```

#### Development Setup
```bash
# Start development environment with hot reloading
docker-compose --profile dev up -d sentiment-dev

# Access development server
curl http://localhost:5001/
```

#### With Optional Services
```bash
# Start with Redis caching
docker-compose --profile cache up -d

# Start with PostgreSQL database
docker-compose --profile database up -d

# Start with full monitoring stack
docker-compose -f docker-compose.monitoring.yml up -d
```

### Multi-Stage Build

The Dockerfile uses multi-stage builds for optimized production images:

- **Builder stage**: Compiles dependencies with build tools
- **Production stage**: Minimal runtime image without build dependencies

Benefits:
- Smaller final image size
- Enhanced security (no build tools in production)
- Faster container startup
- Better layer caching

## Environment Configuration

### Required Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `MODEL_PATH` | Path to trained model file | `model.joblib` | No |
| `LOG_LEVEL` | Logging level | `info` | No |
| `FLASK_ENV` | Flask environment | `production` | No |

### Optional Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `REDIS_URL` | Redis connection string | None |
| `DATABASE_URL` | Database connection string | None |
| `PROMETHEUS_METRICS_ENABLED` | Enable Prometheus metrics | `false` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OpenTelemetry endpoint | None |

### Environment Files

Create `.env` file for local configuration:
```bash
# Copy example configuration
cp .env.example .env

# Edit configuration
nano .env
```

## Security Considerations

### Container Security

The production Docker image implements security best practices:

- **Non-root user**: Application runs as `appuser`
- **Minimal base image**: Uses Python slim image
- **Security updates**: Automatically installs security patches
- **Read-only filesystem**: Application files are read-only
- **Health checks**: Built-in container health monitoring
- **Multi-stage builds**: No build tools in production image

### Network Security

- **Non-privileged ports**: Uses port 8080 (non-privileged)
- **Host binding**: Configurable host binding (default: all interfaces)
- **TLS termination**: Handled by reverse proxy (recommended)

### Secrets Management

**❌ Never include secrets in images:**
- Use environment variables for runtime secrets
- Use Docker secrets for sensitive data
- Mount secret files as volumes

**✅ Recommended approaches:**
```bash
# Using environment variables
docker run -e DATABASE_PASSWORD=secret sentiment-analyzer-pro

# Using Docker secrets
echo "secret_password" | docker secret create db_password -
docker service create --secret db_password sentiment-analyzer-pro

# Using mounted secrets
docker run -v /host/secrets:/app/secrets sentiment-analyzer-pro
```

## Production Deployment

### Docker Swarm

```yaml
# docker-stack.yml
version: '3.8'

services:
  sentiment-analyzer:
    image: sentiment-analyzer-pro:latest
    deploy:
      replicas: 3
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
      resources:
        limits:
          memory: 512M
          cpus: '0.5'
        reservations:
          memory: 256M
          cpus: '0.25'
    ports:
      - "8080:8080"
    networks:
      - sentiment-net
    secrets:
      - model_key
      - db_password

secrets:
  model_key:
    external: true
  db_password:
    external: true

networks:
  sentiment-net:
    external: true
```

Deploy with:
```bash
docker stack deploy -c docker-stack.yml sentiment-stack
```

### Kubernetes

```yaml
# k8s-deployment.yml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sentiment-analyzer
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
        - containerPort: 8080
        env:
        - name: LOG_LEVEL
          value: "info"
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
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /
            port: 8080
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
    targetPort: 8080
  type: LoadBalancer
```

Deploy with:
```bash
kubectl apply -f k8s-deployment.yml
```

## Cloud Deployment

### AWS ECS

1. **Build and push to ECR:**
   ```bash
   # Create ECR repository
   aws ecr create-repository --repository-name sentiment-analyzer-pro

   # Get login token
   aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 123456789012.dkr.ecr.us-west-2.amazonaws.com

   # Build and tag image
   docker build -t sentiment-analyzer-pro .
   docker tag sentiment-analyzer-pro:latest 123456789012.dkr.ecr.us-west-2.amazonaws.com/sentiment-analyzer-pro:latest

   # Push image
   docker push 123456789012.dkr.ecr.us-west-2.amazonaws.com/sentiment-analyzer-pro:latest
   ```

2. **Create ECS task definition and service using AWS Console or CLI**

### Google Cloud Run

```bash
# Build and push to Container Registry
gcloud builds submit --tag gcr.io/PROJECT_ID/sentiment-analyzer-pro

# Deploy to Cloud Run
gcloud run deploy sentiment-analyzer \
  --image gcr.io/PROJECT_ID/sentiment-analyzer-pro \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 512Mi \
  --cpu 1
```

### Azure Container Instances

```bash
# Create resource group
az group create --name sentiment-rg --location eastus

# Create container instance
az container create \
  --resource-group sentiment-rg \
  --name sentiment-analyzer \
  --image sentiment-analyzer-pro:latest \
  --cpu 1 \
  --memory 1 \
  --ports 8080 \
  --environment-variables LOG_LEVEL=info
```

## Monitoring and Observability

### Health Checks

The application includes built-in health checks:

- **Liveness probe**: `GET /`
- **Readiness probe**: `GET /`
- **Metrics endpoint**: `GET /metrics`

### Monitoring Stack

Use the provided monitoring configuration:

```bash
# Start full monitoring stack
docker-compose -f docker-compose.monitoring.yml up -d

# Access monitoring tools
# - Grafana: http://localhost:3000 (admin:admin123)
# - Prometheus: http://localhost:9090
# - Jaeger: http://localhost:16686
```

### Log Management

Structured logging is enabled by default:

```json
{
  "timestamp": "2025-01-01T12:00:00Z",
  "level": "INFO",
  "module": "sentiment_analyzer",
  "message": "Processing prediction request",
  "request_id": "req-123",
  "user_id": "user-456"
}
```

## Performance Tuning

### Resource Limits

Recommended resource allocations:

| Environment | CPU | Memory | Replicas |
|-------------|-----|--------|----------|
| Development | 0.5 cores | 512MB | 1 |
| Staging | 1 core | 1GB | 2 |
| Production | 2 cores | 2GB | 3+ |

### Scaling Considerations

- **Horizontal scaling**: Add more container instances
- **Vertical scaling**: Increase CPU/memory per container
- **Auto-scaling**: Configure based on CPU/memory/request metrics
- **Load balancing**: Use reverse proxy (nginx, HAProxy, cloud LB)

### Caching

Enable Redis caching for improved performance:

```bash
# Start with Redis
docker-compose --profile cache up -d

# Configure caching in environment
export REDIS_URL=redis://localhost:6379
```

## Backup and Recovery

### Model Files

```bash
# Backup trained models
docker run --rm -v model_data:/data -v $(pwd):/backup \
  alpine tar czf /backup/models-backup.tar.gz /data

# Restore models
docker run --rm -v model_data:/data -v $(pwd):/backup \
  alpine tar xzf /backup/models-backup.tar.gz -C /
```

### Database Backups

```bash
# PostgreSQL backup
docker exec postgres_container pg_dump -U sentiment_user sentiment_db > backup.sql

# PostgreSQL restore
docker exec -i postgres_container psql -U sentiment_user sentiment_db < backup.sql
```

## Troubleshooting

### Common Issues

1. **Container fails to start:**
   ```bash
   # Check logs
   docker logs container_name
   
   # Check resource usage
   docker stats container_name
   ```

2. **Application not responding:**
   ```bash
   # Test health endpoint
   curl http://localhost:8080/
   
   # Check container health
   docker inspect container_name | grep Health
   ```

3. **Performance issues:**
   ```bash
   # Monitor resource usage
   docker stats
   
   # Check application metrics
   curl http://localhost:8080/metrics
   ```

### Debug Mode

Enable debug mode for troubleshooting:

```bash
# Development with debug
docker-compose --profile dev up sentiment-dev

# Production with debug logging
docker run -e LOG_LEVEL=debug sentiment-analyzer-pro
```

## CI/CD Integration

### GitHub Actions

The deployment can be automated with GitHub Actions (see workflow documentation).

### GitLab CI

```yaml
# .gitlab-ci.yml
stages:
  - build
  - deploy

build:
  stage: build
  script:
    - docker build -t sentiment-analyzer-pro .
    - docker tag sentiment-analyzer-pro $CI_REGISTRY_IMAGE:latest
    - docker push $CI_REGISTRY_IMAGE:latest

deploy:
  stage: deploy
  script:
    - docker run -d -p 8080:8080 $CI_REGISTRY_IMAGE:latest
  environment:
    name: production
```

This deployment guide ensures reliable, secure, and scalable deployment of the Sentiment Analyzer Pro application across different environments and platforms.