"""Multi-region deployment support for global scalability."""

import os
import json
import requests
import logging
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
import time

logger = logging.getLogger(__name__)

class Region(Enum):
    """Supported deployment regions."""
    US_EAST_1 = "us-east-1"
    US_WEST_2 = "us-west-2"
    EU_WEST_1 = "eu-west-1"
    EU_CENTRAL_1 = "eu-central-1"
    AP_SOUTHEAST_1 = "ap-southeast-1"
    AP_NORTHEAST_1 = "ap-northeast-1"

@dataclass
class RegionConfig:
    """Configuration for a deployment region."""
    region: Region
    endpoint: str
    latency_threshold_ms: float
    max_capacity: int
    current_load: float = 0.0
    health_status: str = "healthy"
    last_health_check: Optional[datetime] = None

class RegionManager:
    """Manages multi-region deployment and load balancing."""
    
    def __init__(self):
        self.regions: Dict[Region, RegionConfig] = {}
        self.primary_region = Region.US_EAST_1
        self.load_balancer_config = self._get_default_load_balancer_config()
        self._initialize_regions()
    
    def _get_default_load_balancer_config(self) -> Dict[str, Any]:
        """Get default load balancer configuration."""
        return {
            "algorithm": "least_connections",
            "health_check_interval": 30,
            "failover_threshold": 0.8,
            "retry_attempts": 3,
            "timeout_seconds": 30
        }
    
    def _initialize_regions(self):
        """Initialize default region configurations."""
        default_configs = {
            Region.US_EAST_1: {
                "endpoint": os.getenv("US_EAST_1_ENDPOINT", "https://sentiment-us-east-1.api.com"),
                "latency_threshold_ms": 50.0,
                "max_capacity": 1000
            },
            Region.US_WEST_2: {
                "endpoint": os.getenv("US_WEST_2_ENDPOINT", "https://sentiment-us-west-2.api.com"),
                "latency_threshold_ms": 75.0,
                "max_capacity": 800
            },
            Region.EU_WEST_1: {
                "endpoint": os.getenv("EU_WEST_1_ENDPOINT", "https://sentiment-eu-west-1.api.com"),
                "latency_threshold_ms": 60.0,
                "max_capacity": 600
            },
            Region.EU_CENTRAL_1: {
                "endpoint": os.getenv("EU_CENTRAL_1_ENDPOINT", "https://sentiment-eu-central-1.api.com"),
                "latency_threshold_ms": 55.0,
                "max_capacity": 500
            },
            Region.AP_SOUTHEAST_1: {
                "endpoint": os.getenv("AP_SOUTHEAST_1_ENDPOINT", "https://sentiment-ap-southeast-1.api.com"),
                "latency_threshold_ms": 80.0,
                "max_capacity": 400
            },
            Region.AP_NORTHEAST_1: {
                "endpoint": os.getenv("AP_NORTHEAST_1_ENDPOINT", "https://sentiment-ap-northeast-1.api.com"),
                "latency_threshold_ms": 70.0,
                "max_capacity": 600
            }
        }
        
        for region, config in default_configs.items():
            self.regions[region] = RegionConfig(
                region=region,
                endpoint=config["endpoint"],
                latency_threshold_ms=config["latency_threshold_ms"],
                max_capacity=config["max_capacity"]
            )
    
    def add_region(self, region_config: RegionConfig):
        """Add a new region configuration."""
        self.regions[region_config.region] = region_config
        logger.info(f"Added region: {region_config.region.value}")
    
    def remove_region(self, region: Region):
        """Remove a region configuration."""
        if region in self.regions:
            del self.regions[region]
            logger.info(f"Removed region: {region.value}")
    
    def get_optimal_region(self, user_location: Optional[Dict[str, float]] = None) -> Region:
        """Get optimal region for user based on location and load."""
        healthy_regions = [
            region for region, config in self.regions.items()
            if config.health_status == "healthy" and 
            config.current_load < self.load_balancer_config["failover_threshold"]
        ]
        
        if not healthy_regions:
            logger.warning("No healthy regions available, using primary region")
            return self.primary_region
        
        if user_location:
            return self._get_geographically_optimal_region(user_location, healthy_regions)
        
        return self._get_load_optimal_region(healthy_regions)
    
    def _get_geographically_optimal_region(
        self, 
        user_location: Dict[str, float], 
        available_regions: List[Region]
    ) -> Region:
        """Get geographically optimal region based on user location."""
        user_lat = user_location.get("latitude", 0.0)
        user_lon = user_location.get("longitude", 0.0)
        
        region_coordinates = {
            Region.US_EAST_1: (39.0458, -76.6413),
            Region.US_WEST_2: (45.5152, -122.6784),
            Region.EU_WEST_1: (53.3498, -6.2603),
            Region.EU_CENTRAL_1: (50.1109, 8.6821),
            Region.AP_SOUTHEAST_1: (1.3521, 103.8198),
            Region.AP_NORTHEAST_1: (35.6762, 139.6503)
        }
        
        min_distance = float('inf')
        optimal_region = available_regions[0]
        
        for region in available_regions:
            if region in region_coordinates:
                reg_lat, reg_lon = region_coordinates[region]
                distance = ((user_lat - reg_lat) ** 2 + (user_lon - reg_lon) ** 2) ** 0.5
                
                load_factor = self.regions[region].current_load
                adjusted_distance = distance * (1 + load_factor)
                
                if adjusted_distance < min_distance:
                    min_distance = adjusted_distance
                    optimal_region = region
        
        return optimal_region
    
    def _get_load_optimal_region(self, available_regions: List[Region]) -> Region:
        """Get region with lowest load."""
        if self.load_balancer_config["algorithm"] == "least_connections":
            return min(available_regions, key=lambda r: self.regions[r].current_load)
        elif self.load_balancer_config["algorithm"] == "round_robin":
            return available_regions[int(time.time()) % len(available_regions)]
        else:
            return available_regions[0]
    
    def health_check_region(self, region: Region) -> bool:
        """Perform health check on a region."""
        config = self.regions.get(region)
        if not config:
            return False
        
        try:
            start_time = time.time()
            response = requests.get(
                f"{config.endpoint}/health",
                timeout=self.load_balancer_config["timeout_seconds"]
            )
            latency = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                config.health_status = "healthy"
                config.last_health_check = datetime.now()
                
                if latency > config.latency_threshold_ms:
                    logger.warning(f"High latency detected in {region.value}: {latency:.2f}ms")
                
                return True
            else:
                config.health_status = "unhealthy"
                logger.error(f"Health check failed for {region.value}: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            config.health_status = "unhealthy"
            logger.error(f"Health check failed for {region.value}: {e}")
            return False
    
    def health_check_all_regions(self) -> Dict[Region, bool]:
        """Perform health check on all regions."""
        results = {}
        for region in self.regions:
            results[region] = self.health_check_region(region)
        return results
    
    def update_region_load(self, region: Region, load: float):
        """Update current load for a region."""
        if region in self.regions:
            self.regions[region].current_load = load
    
    def get_region_stats(self) -> Dict[str, Any]:
        """Get statistics for all regions."""
        stats = {
            "total_regions": len(self.regions),
            "healthy_regions": sum(
                1 for config in self.regions.values()
                if config.health_status == "healthy"
            ),
            "average_load": sum(
                config.current_load for config in self.regions.values()
            ) / len(self.regions) if self.regions else 0,
            "regions": {}
        }
        
        for region, config in self.regions.items():
            stats["regions"][region.value] = {
                "endpoint": config.endpoint,
                "health_status": config.health_status,
                "current_load": config.current_load,
                "max_capacity": config.max_capacity,
                "latency_threshold_ms": config.latency_threshold_ms,
                "last_health_check": config.last_health_check.isoformat() if config.last_health_check else None
            }
        
        return stats
    
    def failover_to_backup(self, failed_region: Region) -> Optional[Region]:
        """Failover to backup region when primary fails."""
        backup_regions = [
            region for region in self.regions
            if region != failed_region and self.regions[region].health_status == "healthy"
        ]
        
        if not backup_regions:
            logger.error("No backup regions available for failover")
            return None
        
        backup_region = self._get_load_optimal_region(backup_regions)
        logger.info(f"Failing over from {failed_region.value} to {backup_region.value}")
        return backup_region

class GlobalLoadBalancer:
    """Global load balancer for sentiment analysis service."""
    
    def __init__(self, region_manager: RegionManager):
        self.region_manager = region_manager
        self.request_counts: Dict[Region, int] = {}
    
    def route_request(
        self, 
        request_data: Dict[str, Any], 
        user_location: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Route request to optimal region."""
        optimal_region = self.region_manager.get_optimal_region(user_location)
        
        if optimal_region not in self.request_counts:
            self.request_counts[optimal_region] = 0
        
        self.request_counts[optimal_region] += 1
        
        # Update load metrics
        current_load = self.request_counts[optimal_region] / self.region_manager.regions[optimal_region].max_capacity
        self.region_manager.update_region_load(optimal_region, current_load)
        
        return {
            "region": optimal_region.value,
            "endpoint": self.region_manager.regions[optimal_region].endpoint,
            "request_id": f"{optimal_region.value}-{int(time.time())}-{self.request_counts[optimal_region]}"
        }
    
    def get_load_balancer_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        total_requests = sum(self.request_counts.values())
        
        return {
            "total_requests": total_requests,
            "request_distribution": {
                region.value: count for region, count in self.request_counts.items()
            },
            "region_stats": self.region_manager.get_region_stats()
        }

_global_region_manager = RegionManager()
_global_load_balancer = GlobalLoadBalancer(_global_region_manager)

def get_region_manager() -> RegionManager:
    """Get global region manager."""
    return _global_region_manager

def get_load_balancer() -> GlobalLoadBalancer:
    """Get global load balancer."""
    return _global_load_balancer

def route_request(
    request_data: Dict[str, Any], 
    user_location: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """Route request globally."""
    return _global_load_balancer.route_request(request_data, user_location)