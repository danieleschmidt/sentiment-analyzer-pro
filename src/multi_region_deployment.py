
import json
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime

class DeploymentRegion(Enum):
    """Supported deployment regions."""
    US_EAST_1 = "us-east-1"
    US_WEST_2 = "us-west-2"
    EU_WEST_1 = "eu-west-1"
    EU_CENTRAL_1 = "eu-central-1"
    AP_SOUTHEAST_1 = "ap-southeast-1"
    AP_NORTHEAST_1 = "ap-northeast-1"
    SA_EAST_1 = "sa-east-1"

@dataclass
class RegionConfig:
    """Configuration for a deployment region."""
    region: DeploymentRegion
    primary: bool
    data_residency_compliant: bool
    applicable_regulations: List[str]
    instance_types: List[str]
    auto_scaling_enabled: bool
    min_instances: int
    max_instances: int
    target_cpu_utilization: float
    backup_region: Optional[DeploymentRegion]

class MultiRegionDeploymentManager:
    """Multi-region deployment management system."""
    
    def __init__(self):
        self.regions = self._initialize_regions()
        self.load_balancing_strategy = "geo_proximity"
        self.failover_strategy = "automatic"
        self.data_replication_strategy = "async"
        
    def _initialize_regions(self) -> Dict[DeploymentRegion, RegionConfig]:
        """Initialize region configurations."""
        
        regions = {
            DeploymentRegion.US_EAST_1: RegionConfig(
                region=DeploymentRegion.US_EAST_1,
                primary=True,
                data_residency_compliant=True,
                applicable_regulations=["CCPA"],
                instance_types=["t3.medium", "t3.large", "c5.xlarge"],
                auto_scaling_enabled=True,
                min_instances=2,
                max_instances=20,
                target_cpu_utilization=70.0,
                backup_region=DeploymentRegion.US_WEST_2
            ),
            DeploymentRegion.US_WEST_2: RegionConfig(
                region=DeploymentRegion.US_WEST_2,
                primary=False,
                data_residency_compliant=True,
                applicable_regulations=["CCPA"],
                instance_types=["t3.medium", "t3.large"],
                auto_scaling_enabled=True,
                min_instances=1,
                max_instances=10,
                target_cpu_utilization=70.0,
                backup_region=DeploymentRegion.US_EAST_1
            ),
            DeploymentRegion.EU_WEST_1: RegionConfig(
                region=DeploymentRegion.EU_WEST_1,
                primary=True,
                data_residency_compliant=True,
                applicable_regulations=["GDPR"],
                instance_types=["t3.medium", "t3.large", "c5.xlarge"],
                auto_scaling_enabled=True,
                min_instances=2,
                max_instances=15,
                target_cpu_utilization=70.0,
                backup_region=DeploymentRegion.EU_CENTRAL_1
            ),
            DeploymentRegion.EU_CENTRAL_1: RegionConfig(
                region=DeploymentRegion.EU_CENTRAL_1,
                primary=False,
                data_residency_compliant=True,
                applicable_regulations=["GDPR"],
                instance_types=["t3.medium"],
                auto_scaling_enabled=True,
                min_instances=1,
                max_instances=8,
                target_cpu_utilization=70.0,
                backup_region=DeploymentRegion.EU_WEST_1
            ),
            DeploymentRegion.AP_SOUTHEAST_1: RegionConfig(
                region=DeploymentRegion.AP_SOUTHEAST_1,
                primary=True,
                data_residency_compliant=True,
                applicable_regulations=["PDPA"],
                instance_types=["t3.medium", "t3.large"],
                auto_scaling_enabled=True,
                min_instances=2,
                max_instances=12,
                target_cpu_utilization=70.0,
                backup_region=DeploymentRegion.AP_NORTHEAST_1
            ),
            DeploymentRegion.AP_NORTHEAST_1: RegionConfig(
                region=DeploymentRegion.AP_NORTHEAST_1,
                primary=False,
                data_residency_compliant=True,
                applicable_regulations=["PIPEDA", "APP"],
                instance_types=["t3.medium"],
                auto_scaling_enabled=True,
                min_instances=1,
                max_instances=8,
                target_cpu_utilization=70.0,
                backup_region=DeploymentRegion.AP_SOUTHEAST_1
            ),
            DeploymentRegion.SA_EAST_1: RegionConfig(
                region=DeploymentRegion.SA_EAST_1,
                primary=True,
                data_residency_compliant=True,
                applicable_regulations=["LGPD"],
                instance_types=["t3.medium"],
                auto_scaling_enabled=True,
                min_instances=1,
                max_instances=6,
                target_cpu_utilization=75.0,
                backup_region=None
            )
        }
        
        return regions
    
    def get_optimal_region(self, client_location: str, compliance_requirements: List[str] = None) -> DeploymentRegion:
        """Get optimal region for client based on location and compliance."""
        
        # Normalize client location
        location_lower = client_location.lower()
        
        # Geographic proximity mapping
        if any(region in location_lower for region in ['usa', 'united states', 'north america']):
            if 'west' in location_lower or any(state in location_lower for state in ['california', 'oregon', 'washington']):
                return DeploymentRegion.US_WEST_2
            return DeploymentRegion.US_EAST_1
        
        elif any(region in location_lower for region in ['europe', 'eu', 'germany', 'france', 'uk']):
            if any(country in location_lower for country in ['germany', 'austria', 'poland']):
                return DeploymentRegion.EU_CENTRAL_1
            return DeploymentRegion.EU_WEST_1
        
        elif any(region in location_lower for region in ['asia', 'singapore', 'malaysia', 'thailand']):
            return DeploymentRegion.AP_SOUTHEAST_1
        
        elif any(region in location_lower for region in ['japan', 'korea', 'taiwan']):
            return DeploymentRegion.AP_NORTHEAST_1
        
        elif any(region in location_lower for region in ['brazil', 'south america', 'latin america']):
            return DeploymentRegion.SA_EAST_1
        
        # Default to US East if no match
        return DeploymentRegion.US_EAST_1
    
    def check_compliance_requirements(self, region: DeploymentRegion, requirements: List[str]) -> bool:
        """Check if region meets compliance requirements."""
        
        if region not in self.regions:
            return False
        
        region_config = self.regions[region]
        region_regulations = set(region_config.applicable_regulations)
        required_regulations = set(requirements)
        
        # Check if all requirements are satisfied
        return required_regulations.issubset(region_regulations)
    
    def get_deployment_manifest(self, region: DeploymentRegion) -> Dict[str, Any]:
        """Generate deployment manifest for a region."""
        
        if region not in self.regions:
            raise ValueError(f"Region {region.value} not supported")
        
        config = self.regions[region]
        
        manifest = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": "sentiment-analyzer-pro",
                "namespace": "default",
                "labels": {
                    "app": "sentiment-analyzer-pro",
                    "region": region.value,
                    "primary": str(config.primary).lower()
                }
            },
            "spec": {
                "replicas": config.min_instances,
                "selector": {
                    "matchLabels": {
                        "app": "sentiment-analyzer-pro"
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": "sentiment-analyzer-pro",
                            "region": region.value
                        }
                    },
                    "spec": {
                        "containers": [{
                            "name": "sentiment-analyzer",
                            "image": "sentiment-analyzer-pro:latest",
                            "ports": [{"containerPort": 5000}],
                            "env": [
                                {"name": "DEPLOYMENT_REGION", "value": region.value},
                                {"name": "PRIMARY_REGION", "value": str(config.primary)},
                                {"name": "APPLICABLE_REGULATIONS", "value": ",".join(config.applicable_regulations)},
                                {"name": "AUTO_SCALING_ENABLED", "value": str(config.auto_scaling_enabled)},
                                {"name": "MIN_INSTANCES", "value": str(config.min_instances)},
                                {"name": "MAX_INSTANCES", "value": str(config.max_instances)}
                            ],
                            "resources": {
                                "requests": {
                                    "cpu": "100m",
                                    "memory": "256Mi"
                                },
                                "limits": {
                                    "cpu": "500m", 
                                    "memory": "512Mi"
                                }
                            },
                            "livenessProbe": {
                                "httpGet": {
                                    "path": "/health",
                                    "port": 5000
                                },
                                "initialDelaySeconds": 30,
                                "periodSeconds": 10
                            },
                            "readinessProbe": {
                                "httpGet": {
                                    "path": "/ready",
                                    "port": 5000
                                },
                                "initialDelaySeconds": 5,
                                "periodSeconds": 5
                            }
                        }],
                        "nodeSelector": {
                            "region": region.value
                        }
                    }
                }
            }
        }
        
        return manifest
    
    def get_auto_scaling_config(self, region: DeploymentRegion) -> Dict[str, Any]:
        """Generate auto-scaling configuration for a region."""
        
        if region not in self.regions:
            raise ValueError(f"Region {region.value} not supported")
        
        config = self.regions[region]
        
        hpa_config = {
            "apiVersion": "autoscaling/v2",
            "kind": "HorizontalPodAutoscaler", 
            "metadata": {
                "name": f"sentiment-analyzer-pro-hpa-{region.value}",
                "namespace": "default"
            },
            "spec": {
                "scaleTargetRef": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": "sentiment-analyzer-pro"
                },
                "minReplicas": config.min_instances,
                "maxReplicas": config.max_instances,
                "metrics": [
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "cpu",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": int(config.target_cpu_utilization)
                            }
                        }
                    },
                    {
                        "type": "Resource", 
                        "resource": {
                            "name": "memory",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": 80
                            }
                        }
                    }
                ],
                "behavior": {
                    "scaleUp": {
                        "stabilizationWindowSeconds": 60,
                        "policies": [{
                            "type": "Percent",
                            "value": 100,
                            "periodSeconds": 60
                        }]
                    },
                    "scaleDown": {
                        "stabilizationWindowSeconds": 300,
                        "policies": [{
                            "type": "Percent",
                            "value": 10,
                            "periodSeconds": 60
                        }]
                    }
                }
            }
        }
        
        return hpa_config
    
    def get_global_load_balancer_config(self) -> Dict[str, Any]:
        """Generate global load balancer configuration."""
        
        # Primary regions for each geographic area
        primary_regions = [region for region, config in self.regions.items() if config.primary]
        
        upstream_servers = []
        for region in primary_regions:
            config = self.regions[region]
            upstream_servers.append({
                "server": f"sentiment-{region.value}.terragon.ai",
                "weight": 100,
                "max_fails": 3,
                "fail_timeout": "30s",
                "backup": False
            })
            
            # Add backup servers
            if config.backup_region and config.backup_region in self.regions:
                upstream_servers.append({
                    "server": f"sentiment-{config.backup_region.value}.terragon.ai",
                    "weight": 50,
                    "max_fails": 2,
                    "fail_timeout": "20s",
                    "backup": True
                })
        
        load_balancer_config = {
            "global_load_balancer": {
                "strategy": self.load_balancing_strategy,
                "health_check_interval": "10s",
                "health_check_timeout": "5s",
                "failover_strategy": self.failover_strategy,
                "upstream_servers": upstream_servers,
                "geographic_routing": {
                    "americas": [DeploymentRegion.US_EAST_1.value, DeploymentRegion.US_WEST_2.value, DeploymentRegion.SA_EAST_1.value],
                    "europe": [DeploymentRegion.EU_WEST_1.value, DeploymentRegion.EU_CENTRAL_1.value],
                    "asia_pacific": [DeploymentRegion.AP_SOUTHEAST_1.value, DeploymentRegion.AP_NORTHEAST_1.value]
                },
                "ssl_termination": True,
                "compression_enabled": True,
                "caching_enabled": True,
                "rate_limiting": {
                    "requests_per_minute": 1000,
                    "burst_size": 200
                }
            }
        }
        
        return load_balancer_config
    
    def generate_deployment_summary(self) -> Dict[str, Any]:
        """Generate deployment summary report."""
        
        total_regions = len(self.regions)
        primary_regions = len([r for r in self.regions.values() if r.primary])
        
        total_min_capacity = sum(config.min_instances for config in self.regions.values())
        total_max_capacity = sum(config.max_instances for config in self.regions.values())
        
        compliance_coverage = {}
        for config in self.regions.values():
            for regulation in config.applicable_regulations:
                if regulation not in compliance_coverage:
                    compliance_coverage[regulation] = []
                compliance_coverage[regulation].append(config.region.value)
        
        summary = {
            "deployment_overview": {
                "total_regions": total_regions,
                "primary_regions": primary_regions,
                "backup_regions": total_regions - primary_regions,
                "global_coverage": True
            },
            "capacity_planning": {
                "total_min_instances": total_min_capacity,
                "total_max_instances": total_max_capacity,
                "auto_scaling_enabled": all(config.auto_scaling_enabled for config in self.regions.values())
            },
            "compliance_coverage": compliance_coverage,
            "high_availability": {
                "multi_region": True,
                "automatic_failover": self.failover_strategy == "automatic",
                "data_replication": self.data_replication_strategy,
                "load_balancing": self.load_balancing_strategy
            },
            "supported_regions": [region.value for region in self.regions.keys()],
            "deployment_date": datetime.now().isoformat()
        }
        
        return summary

# Global multi-region deployment manager
deployment_manager = MultiRegionDeploymentManager()
