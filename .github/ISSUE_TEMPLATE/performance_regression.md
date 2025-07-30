---
name: Performance Regression
about: Report a performance regression in the sentiment analysis pipeline
title: '[PERF] Performance regression in [component]'
labels: performance, regression, priority-high
assignees: ''
---

## Performance Regression Report

### Component Affected
<!-- Select the component experiencing performance degradation -->
- [ ] Model loading
- [ ] Prediction pipeline
- [ ] Text preprocessing  
- [ ] Web API endpoints
- [ ] Batch processing
- [ ] Other: ___

### Performance Metrics
<!-- Provide before/after performance measurements -->

**Before (baseline)**:
- Execution time: 
- Memory usage:
- Throughput:

**After (current)**:
- Execution time:
- Memory usage:
- Throughput:

**Regression percentage**: __%

### Environment Details
- Python version:
- Package version:
- OS:
- Hardware specs:
- Dataset size:

### Reproduction Steps
1. 
2. 
3. 

### Benchmark Commands
```bash
# Commands used to measure performance
```

### Expected Behavior
<!-- What performance was expected based on historical data -->

### Actual Behavior
<!-- What performance is currently observed -->

### Additional Context
<!-- Any additional information about when the regression was first noticed -->

### Profiling Data
<!-- If available, attach profiling output or memory usage graphs -->

### Possible Causes
<!-- Any suspected changes that might have caused the regression -->
- [ ] Recent code changes (PR #xxx)
- [ ] Dependency updates
- [ ] Data changes
- [ ] Environment changes
- [ ] Unknown

---

**Priority Level**: 
- [ ] Critical (>50% regression)
- [ ] High (20-50% regression)  
- [ ] Medium (10-20% regression)
- [ ] Low (<10% regression)