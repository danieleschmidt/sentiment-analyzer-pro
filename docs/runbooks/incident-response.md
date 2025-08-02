# Incident Response Runbook

This runbook provides step-by-step procedures for responding to incidents in the Sentiment Analyzer Pro application.

## Quick Reference

### Emergency Contacts
- **On-call Engineer**: [Your team's on-call rotation]
- **Technical Lead**: [Technical lead contact]
- **Product Owner**: [Product owner contact]

### Critical Dashboards
- **Application Health**: http://localhost:3000/d/app-health
- **Infrastructure Metrics**: http://localhost:3000/d/infrastructure  
- **Error Tracking**: http://localhost:3000/d/errors
- **Performance**: http://localhost:3000/d/performance

### Key URLs
- **Application**: http://localhost:8080
- **Health Check**: http://localhost:8080/health
- **Metrics**: http://localhost:8080/metrics
- **Logs**: docker-compose logs sentiment-analyzer

## Incident Classification

### Severity Levels

#### P0 - Critical (Response: Immediate)
- **Definition**: Complete service outage affecting all users
- **Examples**: 
  - Application completely down
  - 100% error rate
  - Database completely unavailable
- **Response Time**: 15 minutes
- **Escalation**: Immediate to on-call engineer and technical lead

#### P1 - High (Response: 30 minutes)
- **Definition**: Major functionality impacted, affecting significant user base
- **Examples**:
  - High error rate (>25%)
  - Severe performance degradation (>5s response time)
  - Key features unavailable
- **Response Time**: 30 minutes
- **Escalation**: On-call engineer, notify technical lead

#### P2 - Medium (Response: 2 hours)
- **Definition**: Moderate impact, workarounds available
- **Examples**:
  - Elevated error rate (10-25%)
  - Performance degradation (1-5s response time)
  - Non-critical features affected
- **Response Time**: 2 hours
- **Escalation**: Next business day if outside hours

#### P3 - Low (Response: Next business day)
- **Definition**: Minor issues, minimal user impact
- **Examples**:
  - Small error rate increase (5-10%)
  - Minor performance issues
  - Cosmetic problems
- **Response Time**: Next business day
- **Escalation**: None required

## Common Incident Scenarios

### Application Not Responding

#### Symptoms
- Health check endpoint returns 500 or timeout
- No response to API calls
- High response times (>30s)

#### Investigation Steps
1. **Check application status**
   ```bash
   # Check if container is running
   docker ps | grep sentiment-analyzer
   
   # Check application logs
   docker logs sentiment-analyzer --tail 100
   
   # Test health endpoint
   curl -v http://localhost:8080/health
   ```

2. **Check resource usage**
   ```bash
   # Check memory and CPU usage
   docker stats sentiment-analyzer
   
   # Check disk space
   df -h
   
   # Check system load
   top
   ```

3. **Check dependencies**
   ```bash
   # Test database connection (if applicable)
   docker exec postgres pg_isready
   
   # Test Redis connection
   docker exec redis redis-cli ping
   
   # Check network connectivity
   docker network ls
   ```

#### Resolution Actions
- **Immediate**: Restart the application container
  ```bash
  docker-compose restart sentiment-analyzer
  ```
- **If restart fails**: Check logs for specific error messages
- **If resource issue**: Scale up resources or restart with higher limits
- **If dependency issue**: Resolve dependency problem first

### High Error Rate

#### Symptoms
- Error rate >10% for sustained period
- Increase in 4xx/5xx HTTP responses
- Prometheus alert: `HighErrorRate`

#### Investigation Steps
1. **Identify error patterns**
   ```bash
   # Check recent error logs
   docker logs sentiment-analyzer --tail 500 | grep -i error
   
   # Check specific error types
   curl http://localhost:8080/metrics | grep http_requests_total
   ```

2. **Check application metrics**
   - Navigate to Grafana error dashboard
   - Look for error rate by endpoint
   - Check error distribution by response code

3. **Analyze recent changes**
   - Review recent deployments
   - Check for configuration changes
   - Verify model file integrity

#### Resolution Actions
- **If deployment issue**: Rollback to previous version
- **If configuration issue**: Revert configuration changes
- **If model issue**: Restore from backup or retrain
- **If traffic spike**: Scale horizontally

### High Memory Usage

#### Symptoms
- Memory usage >90% sustained
- Out of Memory (OOM) kills in logs
- Prometheus alert: `HighMemoryUsage`

#### Investigation Steps
1. **Check memory usage patterns**
   ```bash
   # Check container memory usage
   docker stats sentiment-analyzer
   
   # Check system memory
   free -h
   
   # Check for memory leaks in logs
   docker logs sentiment-analyzer | grep -i memory
   ```

2. **Identify memory consumers**
   - Check if memory usage correlates with request volume
   - Look for gradual memory increases (potential leak)
   - Check model loading patterns

#### Resolution Actions
- **Immediate**: Restart container to free memory
  ```bash
  docker-compose restart sentiment-analyzer
  ```
- **Short-term**: Increase memory limits
- **Long-term**: Investigate memory leaks, optimize model loading

### Database Connection Issues

#### Symptoms
- Database connection errors in logs
- Timeouts on database operations
- Prometheus alert: `DatabaseConnectionFailed`

#### Investigation Steps
1. **Check database status**
   ```bash
   # Check if database container is running
   docker ps | grep postgres
   
   # Test database connectivity
   docker exec postgres pg_isready -U sentiment_user
   
   # Check database logs
   docker logs postgres --tail 100
   ```

2. **Check connection pool**
   ```bash
   # Check active connections
   docker exec postgres psql -U sentiment_user -d sentiment_db -c "SELECT count(*) FROM pg_stat_activity;"
   
   # Check for long-running queries
   docker exec postgres psql -U sentiment_user -d sentiment_db -c "SELECT pid, now() - pg_stat_activity.query_start AS duration, query FROM pg_stat_activity WHERE (now() - pg_stat_activity.query_start) > interval '5 minutes';"
   ```

#### Resolution Actions
- **If database down**: Restart database container
- **If connection limit reached**: Restart application or increase connection limits
- **If disk full**: Clean up logs or increase disk space

### Performance Degradation

#### Symptoms
- Response times >1s for sustained period
- High CPU usage
- Prometheus alert: `HighResponseTime`

#### Investigation Steps
1. **Check performance metrics**
   - Review response time percentiles in Grafana
   - Check request rate and patterns
   - Analyze slow endpoints

2. **Check resource utilization**
   ```bash
   # Check CPU usage
   docker stats sentiment-analyzer
   
   # Check I/O wait
   iostat -x 1 5
   
   # Check network usage
   iftop
   ```

3. **Check application profiling**
   - Enable debug logging if available
   - Check for expensive operations in logs
   - Review model inference times

#### Resolution Actions
- **If CPU bound**: Scale horizontally or vertically
- **If I/O bound**: Optimize database queries, add caching
- **If model inference slow**: Optimize model or use faster hardware

## Escalation Procedures

### When to Escalate
- Unable to resolve P0 incident within 30 minutes
- P1 incident requires additional expertise
- Incident impacts multiple systems
- Root cause analysis needed

### Escalation Contacts
1. **Technical Lead**: For architectural decisions
2. **DevOps Team**: For infrastructure issues
3. **Data Science Team**: For model-related problems
4. **Product Owner**: For business impact decisions

### Escalation Information to Provide
- Incident severity and impact
- Timeline of events
- Steps already taken
- Current status and blockers
- Business impact assessment

## Post-Incident Activities

### Immediate (Within 24 hours)
1. **Service Restoration Confirmation**
   - Verify all systems are functioning normally
   - Monitor key metrics for stability
   - Document temporary fixes applied

2. **Initial Incident Report**
   - Create incident ticket with basic details
   - Notify stakeholders of resolution
   - Schedule post-mortem meeting

### Follow-up (Within 1 week)
1. **Root Cause Analysis**
   - Conduct thorough investigation
   - Identify contributing factors
   - Document timeline of events

2. **Action Items**
   - Create action items to prevent recurrence
   - Assign owners and due dates
   - Update monitoring and alerting if needed

3. **Process Improvements**
   - Review incident response effectiveness
   - Update runbooks based on learnings
   - Improve monitoring coverage

### Documentation Updates
- Update this runbook with new scenarios
- Add new alerts or dashboards discovered
- Share learnings with the team

## Prevention Strategies

### Monitoring Improvements
- **Proactive Alerting**: Set up alerts before thresholds are breached
- **Synthetic Monitoring**: Regular health checks and end-to-end tests
- **Capacity Planning**: Monitor growth trends and plan scaling

### System Resilience
- **Circuit Breakers**: Implement circuit breakers for external dependencies
- **Graceful Degradation**: Design fallback mechanisms
- **Auto-scaling**: Configure automatic scaling based on metrics

### Regular Maintenance
- **Dependency Updates**: Regular security and feature updates
- **Performance Testing**: Regular load testing
- **Disaster Recovery**: Regular backup and recovery testing

## Useful Commands

### Container Management
```bash
# Restart services
docker-compose restart sentiment-analyzer

# View logs with follow
docker-compose logs -f sentiment-analyzer

# Scale horizontally
docker-compose up -d --scale sentiment-analyzer=3

# Check container health
docker inspect sentiment-analyzer | grep -A 5 Health
```

### Monitoring Queries
```bash
# Check error rate
curl -s "http://localhost:9090/api/v1/query?query=rate(http_requests_total{status=~'5..'}[5m])"

# Check response time
curl -s "http://localhost:9090/api/v1/query?query=histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))"

# Check memory usage
curl -s "http://localhost:9090/api/v1/query?query=container_memory_usage_bytes{name='sentiment-analyzer'}"
```

### Log Analysis
```bash
# Filter error logs
docker logs sentiment-analyzer 2>&1 | grep -i error | tail -20

# Check for specific patterns
docker logs sentiment-analyzer 2>&1 | grep -E "(timeout|connection|failed)" | tail -10

# Count error types
docker logs sentiment-analyzer 2>&1 | grep -i error | awk '{print $4}' | sort | uniq -c
```

---

**Remember**: Stay calm, follow the procedures, and don't hesitate to escalate when needed. Document everything for future learning and improvement.