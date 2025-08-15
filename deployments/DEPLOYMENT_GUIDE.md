# Production Deployment Guide

## Overview
This guide covers the complete production deployment of Sentiment Analyzer Pro with multi-region capabilities, auto-scaling, monitoring, and security.

## Prerequisites
- Docker 20.10+
- Kubernetes 1.21+
- kubectl configured
- Prometheus/Grafana for monitoring

## Quick Deployment

### 1. Build Production Image
```bash
./deployments/scripts/build.sh
```

### 2. Deploy to Kubernetes
```bash
export NAMESPACE=production
export RUN_SMOKE_TESTS=true
./deployments/scripts/deploy.sh
```

### 3. Verify Deployment
```bash
kubectl get pods -l app=sentiment-analyzer-pro -n production
curl http://<service-ip>/health
```

## Architecture

### Components
- **Frontend**: Production Flask application with security headers
- **Caching**: Redis for prediction caching
- **Monitoring**: Prometheus + Grafana
- **Auto-scaling**: Horizontal Pod Autoscaler
- **Security**: Network policies, non-root containers

### Performance Specifications
- **Response Time**: < 100ms (95th percentile)
- **Throughput**: > 1000 requests/second
- **Availability**: 99.9% uptime SLA
- **Auto-scaling**: 2-10 replicas based on CPU/memory

## Security Features

### Container Security
- Non-root user execution
- Read-only root filesystem
- Dropped capabilities
- Security context constraints

### Network Security
- Network policies for traffic isolation
- TLS encryption in transit
- Service mesh integration ready

### Application Security
- Rate limiting (100 req/min per IP)
- Input validation and sanitization
- Security headers (HSTS, XSS protection)
- No sensitive data in logs

## Monitoring & Observability

### Metrics
- Request rate and latency
- Error rates by endpoint
- System resource utilization
- Custom business metrics

### Alerting Rules
- High error rate (>10% for 2min)
- High latency (>1s 95th percentile)
- Service down alerts
- Resource exhaustion warnings

### Logs
- Structured JSON logging
- Correlation IDs for tracing
- Security event logging
- Performance metrics

## Disaster Recovery

### Backup Strategy
- Model artifacts stored in S3
- Configuration in Git
- Database backups (if applicable)

### Recovery Procedures
1. Restore from known good state
2. Rollback to previous version
3. Emergency scaling procedures

## Compliance

### Data Protection
- GDPR compliance ready
- Data anonymization
- Audit trail logging
- Right to deletion support

### Industry Standards
- SOC 2 Type II ready
- ISO 27001 compatible
- PCI DSS considerations

## Troubleshooting

### Common Issues
1. **Pods not starting**: Check resource limits
2. **High latency**: Review auto-scaling settings
3. **Service unavailable**: Verify ingress configuration

### Debug Commands
```bash
# Check pod logs
kubectl logs -l app=sentiment-analyzer-pro -n production

# Check resource usage
kubectl top pods -n production

# Check service endpoints
kubectl get endpoints -n production
```

## Maintenance

### Regular Tasks
- Security patch updates
- Model retraining and deployment
- Performance optimization
- Capacity planning

### Upgrade Procedures
1. Test in staging environment
2. Rolling deployment strategy
3. Rollback plan preparation
4. Post-deployment verification

## Support Contacts
- **On-call**: ops-team@company.com
- **Security**: security@company.com
- **ML Team**: ml-platform@company.com
