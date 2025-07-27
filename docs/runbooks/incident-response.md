# Incident Response Runbook

## Overview
This runbook provides step-by-step procedures for responding to incidents in the Sentiment Analyzer Pro system.

## Incident Classification

### Severity Levels

#### P0 - Critical
- Complete service outage
- Data loss or corruption
- Security breach
- **Response Time:** Immediate (< 15 minutes)

#### P1 - High
- Significant feature unavailable
- Performance degradation > 50%
- Multiple user reports
- **Response Time:** < 1 hour

#### P2 - Medium
- Minor feature issues
- Performance degradation < 50%
- Single user reports
- **Response Time:** < 4 hours

#### P3 - Low
- Cosmetic issues
- Documentation errors
- Enhancement requests
- **Response Time:** < 24 hours

## Incident Response Process

### 1. Detection and Alert
- **Monitoring alerts** via Prometheus/Grafana
- **User reports** via GitHub issues or support channels
- **Health check failures** in CI/CD pipeline

### 2. Initial Response (First 5 minutes)
1. **Acknowledge the incident**
   - Update incident status
   - Assign incident commander
   
2. **Assess severity**
   - Determine impact scope
   - Classify incident priority
   
3. **Initial communication**
   - Notify relevant stakeholders
   - Create incident tracking issue

### 3. Investigation and Diagnosis

#### Quick Health Checks
```bash
# Check service status
curl -f https://sentiment-analyzer.example.com/health

# Check container status
docker ps | grep sentiment-analyzer

# Check logs
docker logs sentiment-analyzer --tail=100

# Check resource usage
docker stats sentiment-analyzer
```

#### System Diagnostics
```bash
# Check application logs
tail -f /app/logs/sentiment_analyzer.log

# Check error rates
grep "ERROR" /app/logs/sentiment_analyzer.log | tail -20

# Check performance metrics
curl https://sentiment-analyzer.example.com/metrics
```

### 4. Immediate Mitigation

#### Common Mitigation Strategies

**Service Restart**
```bash
# Restart service
docker restart sentiment-analyzer

# Or with docker-compose
docker-compose restart sentiment-analyzer
```

**Rollback Deployment**
```bash
# Kubernetes rollback
kubectl rollout undo deployment/sentiment-analyzer

# Docker Swarm rollback
docker service rollback sentiment-analyzer
```

**Scale Resources**
```bash
# Scale up replicas
kubectl scale deployment sentiment-analyzer --replicas=5

# Increase memory limits
kubectl patch deployment sentiment-analyzer -p '{"spec":{"template":{"spec":{"containers":[{"name":"sentiment-analyzer","resources":{"limits":{"memory":"2Gi"}}}]}}}}'
```

### 5. Communication Templates

#### Initial Incident Report
```
**INCIDENT ALERT - [SEVERITY]**

**Summary:** Brief description of the issue
**Impact:** Who/what is affected
**Status:** Under investigation
**ETA:** Estimated resolution time
**Updates:** Will provide updates every 30 minutes

**Current Actions:**
- [ ] Investigating root cause
- [ ] Implementing immediate mitigation
- [ ] Monitoring system health
```

#### Status Update
```
**INCIDENT UPDATE - [SEVERITY]**

**Summary:** Updated description
**Status:** [Investigating/Mitigating/Resolved]
**Progress:** What has been done
**Next Steps:** Planned actions
**ETA:** Updated estimate
```

#### Resolution Notice
```
**INCIDENT RESOLVED - [SEVERITY]**

**Summary:** Final description
**Root Cause:** What caused the incident
**Resolution:** How it was fixed
**Duration:** Total incident time
**Follow-up:** Post-incident actions planned
```

## Common Incidents and Solutions

### Application Won't Start

**Symptoms:**
- Container exits immediately
- Health checks failing
- "Connection refused" errors

**Investigation:**
```bash
# Check container logs
docker logs sentiment-analyzer

# Check configuration
docker exec sentiment-analyzer env | grep -E "^(FLASK|MODEL|LOG)"

# Check file permissions
docker exec sentiment-analyzer ls -la /app/models/
```

**Common Causes:**
- Missing model files
- Invalid environment variables
- Port conflicts
- Insufficient permissions

**Solutions:**
```bash
# Fix missing model file
docker cp ./models/sentiment_model.joblib sentiment-analyzer:/app/models/

# Update environment variables
docker exec -e MODEL_PATH=/app/models/sentiment_model.joblib sentiment-analyzer

# Fix permissions
docker exec sentiment-analyzer chmod 644 /app/models/*
```

### High Memory Usage

**Symptoms:**
- OOMKilled containers
- Slow response times
- High swap usage

**Investigation:**
```bash
# Check memory usage
docker stats sentiment-analyzer

# Check process memory
docker exec sentiment-analyzer ps aux --sort=-%mem

# Check model sizes
docker exec sentiment-analyzer du -sh /app/models/*
```

**Solutions:**
```bash
# Increase memory limits
kubectl patch deployment sentiment-analyzer -p '{"spec":{"template":{"spec":{"containers":[{"name":"sentiment-analyzer","resources":{"limits":{"memory":"4Gi"}}}]}}}}'

# Enable model caching optimizations
docker exec sentiment-analyzer -e ENABLE_MODEL_CACHING=true
```

### Database Connection Issues

**Symptoms:**
- Database connection errors
- Slow query performance
- Connection timeouts

**Investigation:**
```bash
# Check database connectivity
docker exec sentiment-analyzer python -c "import psycopg2; conn = psycopg2.connect('postgresql://user:pass@db:5432/sentiment_db')"

# Check connection pool
docker exec sentiment-analyzer netstat -an | grep 5432
```

**Solutions:**
```bash
# Restart database connection
docker restart sentiment-analyzer

# Check database health
docker exec postgres pg_isready -U sentiment_user
```

### Model Prediction Errors

**Symptoms:**
- 500 errors on /predict endpoint
- Model loading failures
- Inconsistent predictions

**Investigation:**
```bash
# Test model manually
docker exec sentiment-analyzer python -c "
import joblib
model = joblib.load('/app/models/sentiment_model.joblib')
print(model.predict(['test text']))
"

# Check model file integrity
docker exec sentiment-analyzer ls -la /app/models/
docker exec sentiment-analyzer file /app/models/sentiment_model.joblib
```

**Solutions:**
```bash
# Reload model
docker exec sentiment-analyzer pkill -f "python.*webapp"

# Replace corrupted model
docker cp ./backup/sentiment_model.joblib sentiment-analyzer:/app/models/
```

## Post-Incident Procedures

### 1. Root Cause Analysis
- Document timeline of events
- Identify contributing factors
- Determine preventive measures

### 2. Post-Incident Review
- Schedule review meeting within 48 hours
- Include all involved parties
- Document lessons learned

### 3. Action Items
- Create GitHub issues for improvements
- Update runbooks and documentation
- Implement monitoring improvements

### 4. Follow-up
- Monitor system for recurring issues
- Validate implemented fixes
- Update incident response procedures

## Emergency Contacts

### On-Call Rotation
- **Primary:** [Name] - [Phone] - [Email]
- **Secondary:** [Name] - [Phone] - [Email]
- **Escalation:** [Name] - [Phone] - [Email]

### External Dependencies
- **Cloud Provider:** [Support Contact]
- **CDN Provider:** [Support Contact]
- **Database Provider:** [Support Contact]

## Tools and Resources

### Monitoring Dashboards
- **Application Metrics:** https://grafana.example.com/d/app-metrics
- **Infrastructure Metrics:** https://grafana.example.com/d/infra-metrics
- **Error Tracking:** https://sentry.example.com/sentiment-analyzer

### Log Aggregation
- **Application Logs:** https://kibana.example.com/app/discover
- **System Logs:** https://logs.example.com/sentiment-analyzer

### Communication Channels
- **Incident Channel:** #incidents-sentiment-analyzer
- **Team Channel:** #sentiment-analyzer-team
- **Status Page:** https://status.example.com

## Quick Reference Commands

### Health Checks
```bash
# Application health
curl -f https://sentiment-analyzer.example.com/health

# Detailed metrics
curl https://sentiment-analyzer.example.com/metrics

# Database connectivity
docker exec sentiment-analyzer python -c "from src.config import get_db_connection; get_db_connection()"
```

### Log Analysis
```bash
# Recent errors
docker logs sentiment-analyzer 2>&1 | grep ERROR | tail -20

# Performance issues
docker logs sentiment-analyzer 2>&1 | grep -E "(slow|timeout|performance)" | tail -10

# Security events
docker logs sentiment-analyzer 2>&1 | grep -E "(401|403|security)" | tail -10
```

### System Recovery
```bash
# Graceful restart
docker exec sentiment-analyzer pkill -TERM python
docker restart sentiment-analyzer

# Force restart
docker kill sentiment-analyzer
docker start sentiment-analyzer

# Emergency rollback
kubectl rollout undo deployment/sentiment-analyzer --to-revision=1
```