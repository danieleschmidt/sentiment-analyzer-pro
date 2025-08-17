# Sentiment Analyzer Pro - Production Deployment

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
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
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
