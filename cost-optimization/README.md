# Cost Optimization Guide

This guide provides strategies and automation for optimizing operational costs across different deployment scenarios while maintaining performance and reliability.

## Cost Analysis Framework

### 1. Cost Categories

| Category | Components | Optimization Strategy |
|----------|------------|----------------------|
| **Compute** | CPU, Memory, GPU | Right-sizing, auto-scaling |
| **Storage** | Model files, data, logs | Compression, lifecycle policies |
| **Network** | API calls, data transfer | Caching, CDN, batching |
| **External Services** | ML APIs, monitoring tools | Usage optimization, alternatives |

### 2. Cost Monitoring

```python
# Cost tracking integration
class CostTracker:
    def __init__(self):
        self.costs = {
            'compute_hours': 0,
            'api_calls': 0,
            'storage_gb': 0,
            'network_gb': 0
        }
    
    def track_compute_usage(self, hours: float, instance_type: str):
        cost_per_hour = {
            'small': 0.05,
            'medium': 0.10,
            'large': 0.20,
            'gpu': 0.90
        }
        self.costs['compute_hours'] += hours * cost_per_hour.get(instance_type, 0.10)
    
    def track_api_usage(self, calls: int):
        cost_per_1k_calls = 0.002
        self.costs['api_calls'] += (calls / 1000) * cost_per_1k_calls
    
    def get_monthly_estimate(self):
        return sum(self.costs.values()) * 30  # Daily to monthly
```

## Compute Optimization

### 1. Right-sizing Strategy
```yaml
# container-sizing.yml
apiVersion: v1
kind: ConfigMap
metadata:
  name: resource-profiles
data:
  development: |
    resources:
      requests:
        cpu: 100m
        memory: 128Mi
      limits:
        cpu: 500m
        memory: 512Mi
  
  production-light: |
    resources:
      requests:
        cpu: 200m
        memory: 256Mi
      limits:
        cpu: 1000m
        memory: 1Gi
  
  production-heavy: |
    resources:
      requests:
        cpu: 500m
        memory: 512Mi
      limits:
        cpu: 2000m
        memory: 2Gi
```

### 2. Auto-scaling Configuration
```python
# Smart auto-scaling based on cost and performance
class CostAwareAutoScaler:
    def __init__(self):
        self.cost_threshold = 10.0  # USD per hour
        self.performance_threshold = 500  # ms response time
    
    def should_scale_up(self, current_cost: float, avg_response_time: float, queue_length: int):
        # Scale up if performance is degraded but cost allows
        if avg_response_time > self.performance_threshold and current_cost < self.cost_threshold:
            return True
        
        # Emergency scaling for queue buildup
        if queue_length > 100:
            return True
        
        return False
    
    def should_scale_down(self, current_cost: float, avg_response_time: float, cpu_usage: float):
        # Scale down if over-provisioned
        if cpu_usage < 30 and avg_response_time < self.performance_threshold * 0.5:
            return True
        
        # Cost-driven scaling
        if current_cost > self.cost_threshold:
            return True
        
        return False
```

### 3. Spot Instance Usage
```bash
#!/bin/bash
# spot-instance-deployment.sh

# Use spot instances for non-critical workloads
kubectl apply -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  labels:
    cost-optimization: spot-instance
spec:
  nodeSelector:
    node-type: spot
  tolerations:
  - key: spot-instance
    operator: Equal
    value: "true"
    effect: NoSchedule
  containers:
  - name: sentiment-analyzer
    image: sentiment-analyzer:latest
    resources:
      requests:
        cpu: 100m
        memory: 256Mi
EOF
```

## Storage Optimization

### 1. Model Compression
```python
# Model size optimization
import joblib
import gzip
import pickle

class ModelCompressor:
    @staticmethod
    def compress_model(model, filepath: str, compression_level: int = 6):
        """Compress model files to reduce storage costs."""
        with gzip.open(f"{filepath}.gz", 'wb', compresslevel=compression_level) as f:
            pickle.dump(model, f)
        
        # Compare sizes
        original_size = os.path.getsize(filepath) if os.path.exists(filepath) else 0
        compressed_size = os.path.getsize(f"{filepath}.gz")
        
        savings = ((original_size - compressed_size) / original_size) * 100
        print(f"Storage savings: {savings:.1f}% ({original_size} -> {compressed_size} bytes)")
    
    @staticmethod
    def load_compressed_model(filepath: str):
        """Load compressed model."""
        with gzip.open(f"{filepath}.gz", 'rb') as f:
            return pickle.load(f)
```

### 2. Data Lifecycle Management
```yaml
# storage-lifecycle.yml
apiVersion: v1
kind: ConfigMap
metadata:
  name: storage-lifecycle-policy
data:
  policy.json: |
    {
      "rules": [
        {
          "name": "transition-to-ia",
          "filter": {"prefix": "logs/"},
          "status": "Enabled",
          "transitions": [
            {
              "days": 30,
              "storage_class": "STANDARD_IA"
            },
            {
              "days": 90,
              "storage_class": "GLACIER"
            }
          ]
        },
        {
          "name": "delete-old-models",
          "filter": {"prefix": "models/archived/"},
          "status": "Enabled",
          "expiration": {"days": 365}
        }
      ]
    }
```

### 3. Efficient Data Formats
```python
# Data format optimization for cost and performance
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

def optimize_data_storage(df: pd.DataFrame, output_path: str):
    """Convert CSV to Parquet for better compression and faster reads."""
    
    # Optimize data types
    for col in df.select_dtypes(include=['object']):
        if df[col].nunique() < len(df) * 0.5:  # Categorical if < 50% unique
            df[col] = df[col].astype('category')
    
    # Write to Parquet with compression
    df.to_parquet(
        output_path,
        engine='pyarrow',
        compression='snappy',
        index=False
    )
    
    # Compare sizes
    csv_size = df.memory_usage(deep=True).sum()
    parquet_size = os.path.getsize(output_path)
    
    savings = ((csv_size - parquet_size) / csv_size) * 100
    print(f"Storage optimization: {savings:.1f}% reduction")
```

## Network Cost Optimization

### 1. Response Caching
```python
# Intelligent caching to reduce API calls and network costs
from functools import lru_cache
import hashlib
import time

class CostOptimizedCache:
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size
        self.ttl = ttl
    
    def get_cache_key(self, text: str) -> str:
        """Generate cache key for text input."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def get(self, key: str):
        """Get cached result if valid."""
        if key in self.cache:
            # Check TTL
            if time.time() - self.access_times[key] < self.ttl:
                return self.cache[key]
            else:
                # Expired
                del self.cache[key]
                del self.access_times[key]
        return None
    
    def set(self, key: str, value):
        """Cache result with cost-aware eviction."""
        # Evict if at capacity
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = min(self.access_times, key=self.access_times.get)
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
        
        self.cache[key] = value
        self.access_times[key] = time.time()
```

### 2. Request Batching
```python
# Batch requests to reduce API overhead
import asyncio
from typing import List

class BatchProcessor:
    def __init__(self, batch_size: int = 50, max_wait_time: float = 1.0):
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.pending_requests = []
        self.last_batch_time = time.time()
    
    async def add_request(self, request_data):
        """Add request to batch queue."""
        self.pending_requests.append(request_data)
        
        # Process batch if conditions met
        should_process = (
            len(self.pending_requests) >= self.batch_size or
            time.time() - self.last_batch_time > self.max_wait_time
        )
        
        if should_process:
            return await self.process_batch()
        
        return None
    
    async def process_batch(self) -> List:
        """Process batched requests efficiently."""
        if not self.pending_requests:
            return []
        
        batch = self.pending_requests.copy()
        self.pending_requests.clear()
        self.last_batch_time = time.time()
        
        # Batch API call (more cost-effective)
        results = await self.batch_predict(batch)
        
        return results
    
    async def batch_predict(self, texts: List[str]) -> List:
        """Efficient batch prediction."""
        # Implementation depends on model
        pass
```

## External Service Cost Management

### 1. API Usage Optimization
```python
# Smart API usage to minimize external service costs
class APIUsageOptimizer:
    def __init__(self):
        self.daily_quota = 10000  # API calls per day
        self.current_usage = 0
        self.rate_limits = {
            'premium': 1000,  # calls per hour
            'standard': 100,
            'basic': 10
        }
    
    def should_use_external_api(self, confidence_threshold: float = 0.8) -> bool:
        """Decide whether to use external API based on cost and confidence."""
        # Use local model first
        local_confidence = self.get_local_prediction_confidence()
        
        # Only use external API if local confidence is low
        if local_confidence > confidence_threshold:
            return False
        
        # Check quota
        if self.current_usage >= self.daily_quota:
            return False
        
        return True
    
    def get_local_prediction_confidence(self) -> float:
        """Get confidence score from local model."""
        # Implementation specific to your models
        return 0.75  # Example
```

### 2. Service Tier Management
```python
# Dynamic service tier selection based on requirements
class ServiceTierManager:
    def __init__(self):
        self.tiers = {
            'basic': {'cost': 0.001, 'latency': 500, 'accuracy': 0.85},
            'standard': {'cost': 0.005, 'latency': 200, 'accuracy': 0.90},
            'premium': {'cost': 0.020, 'latency': 50, 'accuracy': 0.95}
        }
    
    def select_tier(self, priority: str, budget_limit: float) -> str:
        """Select appropriate service tier based on requirements."""
        if priority == 'cost':
            # Choose cheapest option within budget
            for tier, config in sorted(self.tiers.items(), key=lambda x: x[1]['cost']):
                if config['cost'] <= budget_limit:
                    return tier
        
        elif priority == 'performance':
            # Choose fastest option within budget
            for tier, config in sorted(self.tiers.items(), key=lambda x: x[1]['latency']):
                if config['cost'] <= budget_limit:
                    return tier
        
        elif priority == 'accuracy':
            # Choose most accurate option within budget
            for tier, config in sorted(self.tiers.items(), key=lambda x: x[1]['accuracy'], reverse=True):
                if config['cost'] <= budget_limit:
                    return tier
        
        return 'basic'  # Fallback
```

## Cost Alerting and Monitoring

### 1. Cost Threshold Alerts
```python
# Automated cost monitoring and alerting
class CostMonitor:
    def __init__(self, daily_budget: float = 50.0):
        self.daily_budget = daily_budget
        self.current_spend = 0.0
        self.alert_thresholds = [0.5, 0.8, 0.9, 1.0]  # 50%, 80%, 90%, 100%
        self.alerts_sent = set()
    
    def update_spend(self, amount: float):
        """Update current spending and check thresholds."""
        self.current_spend += amount
        self.check_thresholds()
    
    def check_thresholds(self):
        """Check if any alert thresholds have been crossed."""
        spend_ratio = self.current_spend / self.daily_budget
        
        for threshold in self.alert_thresholds:
            if spend_ratio >= threshold and threshold not in self.alerts_sent:
                self.send_cost_alert(threshold, spend_ratio)
                self.alerts_sent.add(threshold)
    
    def send_cost_alert(self, threshold: float, current_ratio: float):
        """Send cost alert notification."""
        message = f"""
        Cost Alert: {threshold * 100}% of daily budget reached
        Current spend: ${self.current_spend:.2f} / ${self.daily_budget:.2f}
        Percentage: {current_ratio * 100:.1f}%
        """
        # Send via your preferred notification method
        print(message)  # Replace with actual notification
```

### 2. Cost Optimization Recommendations
```python
# Automated cost optimization recommendations
class CostOptimizer:
    def analyze_usage_patterns(self, usage_data: dict) -> List[str]:
        """Analyze usage patterns and provide cost optimization recommendations."""
        recommendations = []
        
        # CPU utilization analysis
        if usage_data.get('avg_cpu_utilization', 0) < 30:
            recommendations.append(
                "Consider downgrading instance size - CPU utilization is consistently low"
            )
        
        # Memory usage analysis
        if usage_data.get('avg_memory_utilization', 0) < 40:
            recommendations.append(
                "Memory is under-utilized - consider smaller instance or memory-optimized type"
            )
        
        # API usage patterns
        api_usage = usage_data.get('api_calls_per_hour', [])
        if api_usage and max(api_usage) / min(api_usage) > 5:
            recommendations.append(
                "API usage varies significantly - consider implementing request queuing"
            )
        
        # Storage analysis
        if usage_data.get('storage_growth_rate', 0) > 0.1:  # 10% per day
            recommendations.append(
                "Storage growing rapidly - implement data lifecycle policies"
            )
        
        return recommendations
```

## Implementation Checklist

### Daily Operations
- [ ] Monitor cost dashboards
- [ ] Review resource utilization metrics
- [ ] Check cost alerts and thresholds
- [ ] Analyze API usage patterns

### Weekly Reviews
- [ ] Review cost optimization recommendations
- [ ] Analyze usage trends and patterns
- [ ] Update resource allocation if needed
- [ ] Review external service usage

### Monthly Optimization
- [ ] Comprehensive cost analysis
- [ ] Update cost optimization strategies
- [ ] Review and adjust budgets
- [ ] Implement new cost-saving measures

## Cost Optimization ROI

Track the return on investment for cost optimization efforts:

```python
def calculate_optimization_roi(baseline_cost: float, optimized_cost: float, 
                             implementation_effort_hours: float, hourly_rate: float = 100) -> dict:
    """Calculate ROI for cost optimization initiatives."""
    
    monthly_savings = baseline_cost - optimized_cost
    implementation_cost = implementation_effort_hours * hourly_rate
    
    # Break-even calculation
    break_even_months = implementation_cost / monthly_savings if monthly_savings > 0 else float('inf')
    
    # Annual ROI
    annual_savings = monthly_savings * 12
    roi_percentage = ((annual_savings - implementation_cost) / implementation_cost) * 100
    
    return {
        'monthly_savings': monthly_savings,
        'annual_savings': annual_savings,
        'implementation_cost': implementation_cost,
        'break_even_months': break_even_months,
        'roi_percentage': roi_percentage
    }
```

For detailed cost tracking and automation scripts, see the `/scripts/cost-optimization/` directory.