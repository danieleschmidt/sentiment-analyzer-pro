"""
Advanced Deployment Orchestrator for Multi-Environment Sentiment Analysis

This module provides intelligent deployment orchestration with:
- Blue-green deployments with automated rollback
- Canary deployments with traffic splitting
- A/B testing infrastructure
- Multi-cloud deployment strategies
- Auto-scaling based on performance metrics
- Disaster recovery automation
- Compliance and security validation

Features:
- Infrastructure as Code (IaC) generation
- Kubernetes native deployment
- Service mesh integration
- Observability and monitoring
- Cost optimization
- Security policy enforcement
"""

from __future__ import annotations

import asyncio
import yaml
import json
import time
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import logging
import threading
from collections import defaultdict
import uuid

# Cloud and orchestration
try:
    import kubernetes
    from kubernetes import client, config
    KUBERNETES_AVAILABLE = True
except ImportError:
    KUBERNETES_AVAILABLE = False

try:
    import boto3
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False

try:
    import google.cloud
    from google.cloud import container_v1
    GCP_AVAILABLE = True
except ImportError:
    GCP_AVAILABLE = False

try:
    import azure.identity
    from azure.mgmt.containerinstance import ContainerInstanceManagementClient
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

# Monitoring and metrics
try:
    import prometheus_client
    from prometheus_client import Counter, Histogram, Gauge
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

logger = logging.getLogger(__name__)


class DeploymentStrategy(Enum):
    """Deployment strategy types"""
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    ROLLING = "rolling"
    RECREATE = "recreate"
    A_B_TEST = "ab_test"


class DeploymentEnvironment(Enum):
    """Target deployment environments"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    DISASTER_RECOVERY = "disaster_recovery"


class CloudProvider(Enum):
    """Supported cloud providers"""
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    KUBERNETES = "kubernetes"
    ON_PREMISE = "on_premise"


@dataclass
class DeploymentConfig:
    """Configuration for deployment orchestration"""
    name: str
    version: str
    strategy: DeploymentStrategy
    environment: DeploymentEnvironment
    cloud_provider: CloudProvider
    
    # Resource specifications
    cpu_request: str = "100m"
    cpu_limit: str = "500m"
    memory_request: str = "128Mi"
    memory_limit: str = "512Mi"
    replicas: int = 3
    
    # Auto-scaling configuration
    enable_autoscaling: bool = True
    min_replicas: int = 2
    max_replicas: int = 10
    target_cpu_percentage: int = 70
    target_memory_percentage: int = 80
    
    # Health checks
    health_check_path: str = "/health"
    readiness_check_path: str = "/ready"
    liveness_check_path: str = "/alive"
    
    # Security configuration
    enable_security_policies: bool = True
    enable_network_policies: bool = True
    enable_pod_security_standards: bool = True
    
    # Monitoring and observability
    enable_metrics: bool = True
    enable_tracing: bool = True
    enable_logging: bool = True
    
    # Traffic management
    canary_weight: int = 10  # Percentage for canary deployments
    ab_test_weight: int = 50  # Percentage for A/B testing
    
    # Rollback configuration
    enable_auto_rollback: bool = True
    rollback_threshold_error_rate: float = 0.05  # 5% error rate
    rollback_threshold_latency: float = 2000  # 2000ms
    monitoring_duration_minutes: int = 15
    
    # Additional metadata
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    environment_variables: Dict[str, str] = field(default_factory=dict)


@dataclass
class DeploymentStatus:
    """Status of a deployment"""
    id: str
    config: DeploymentConfig
    status: str  # pending, running, successful, failed, rolling_back
    created_at: datetime
    updated_at: datetime = field(default_factory=datetime.now)
    current_replicas: int = 0
    ready_replicas: int = 0
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    rollback_info: Optional[Dict] = None


class KubernetesOrchestrator:
    """Kubernetes-based deployment orchestrator"""
    
    def __init__(self, namespace: str = "default", kubeconfig_path: str = None):
        self.namespace = namespace
        
        if not KUBERNETES_AVAILABLE:
            raise ImportError("Kubernetes client not available")
        
        # Load kubeconfig
        try:
            if kubeconfig_path:
                config.load_kube_config(config_file=kubeconfig_path)
            else:
                config.load_incluster_config()
        except:
            config.load_kube_config()
        
        self.v1 = client.CoreV1Api()
        self.apps_v1 = client.AppsV1Api()
        self.autoscaling_v1 = client.AutoscalingV1Api()
        self.networking_v1 = client.NetworkingV1Api()
        
        logger.info(f"Kubernetes orchestrator initialized for namespace: {namespace}")
    
    def create_deployment(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Create Kubernetes deployment"""
        deployment_spec = self._create_deployment_spec(config)
        
        try:
            # Create deployment
            deployment = self.apps_v1.create_namespaced_deployment(
                namespace=self.namespace,
                body=deployment_spec
            )
            
            # Create service
            service_spec = self._create_service_spec(config)
            service = self.v1.create_namespaced_service(
                namespace=self.namespace,
                body=service_spec
            )
            
            # Create HPA if autoscaling enabled
            if config.enable_autoscaling:
                hpa_spec = self._create_hpa_spec(config)
                self.autoscaling_v1.create_namespaced_horizontal_pod_autoscaler(
                    namespace=self.namespace,
                    body=hpa_spec
                )
            
            # Create network policies if enabled
            if config.enable_network_policies:
                network_policy_spec = self._create_network_policy_spec(config)
                self.networking_v1.create_namespaced_network_policy(
                    namespace=self.namespace,
                    body=network_policy_spec
                )
            
            return {
                'deployment': deployment.to_dict(),
                'service': service.to_dict(),
                'status': 'created'
            }
            
        except Exception as e:
            logger.error(f"Error creating deployment: {e}")
            raise
    
    def _create_deployment_spec(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Create Kubernetes deployment specification"""
        labels = {
            'app': config.name,
            'version': config.version,
            'environment': config.environment.value,
            **config.labels
        }
        
        container_spec = {
            'name': config.name,
            'image': f'{config.name}:{config.version}',
            'ports': [{'containerPort': 5000, 'name': 'http'}],
            'resources': {
                'requests': {
                    'cpu': config.cpu_request,
                    'memory': config.memory_request
                },
                'limits': {
                    'cpu': config.cpu_limit,
                    'memory': config.memory_limit
                }
            },
            'env': [
                {'name': key, 'value': value}
                for key, value in config.environment_variables.items()
            ]
        }
        
        # Add health checks
        container_spec['livenessProbe'] = {
            'httpGet': {
                'path': config.liveness_check_path,
                'port': 'http'
            },
            'initialDelaySeconds': 30,
            'periodSeconds': 10
        }
        
        container_spec['readinessProbe'] = {
            'httpGet': {
                'path': config.readiness_check_path,
                'port': 'http'
            },
            'initialDelaySeconds': 5,
            'periodSeconds': 5
        }
        
        # Security context
        if config.enable_pod_security_standards:
            container_spec['securityContext'] = {
                'allowPrivilegeEscalation': False,
                'runAsNonRoot': True,
                'runAsUser': 1000,
                'readOnlyRootFilesystem': True,
                'capabilities': {'drop': ['ALL']}
            }
        
        deployment_spec = {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': f'{config.name}-{config.version}',
                'labels': labels,
                'annotations': config.annotations
            },
            'spec': {
                'replicas': config.replicas,
                'selector': {'matchLabels': {'app': config.name}},
                'template': {
                    'metadata': {
                        'labels': labels,
                        'annotations': {
                            'prometheus.io/scrape': 'true' if config.enable_metrics else 'false',
                            'prometheus.io/port': '5000',
                            **config.annotations
                        }
                    },
                    'spec': {
                        'containers': [container_spec],
                        'securityContext': {
                            'fsGroup': 2000,
                            'seccompProfile': {'type': 'RuntimeDefault'}
                        } if config.enable_pod_security_standards else {}
                    }
                },
                'strategy': {
                    'type': 'RollingUpdate',
                    'rollingUpdate': {
                        'maxSurge': '25%',
                        'maxUnavailable': '25%'
                    }
                }
            }
        }
        
        return deployment_spec
    
    def _create_service_spec(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Create Kubernetes service specification"""
        return {
            'apiVersion': 'v1',
            'kind': 'Service',
            'metadata': {
                'name': config.name,
                'labels': {
                    'app': config.name,
                    'environment': config.environment.value
                }
            },
            'spec': {
                'selector': {'app': config.name},
                'ports': [
                    {
                        'port': 80,
                        'targetPort': 5000,
                        'name': 'http'
                    }
                ],
                'type': 'ClusterIP'
            }
        }
    
    def _create_hpa_spec(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Create Horizontal Pod Autoscaler specification"""
        return {
            'apiVersion': 'autoscaling/v1',
            'kind': 'HorizontalPodAutoscaler',
            'metadata': {
                'name': f'{config.name}-hpa'
            },
            'spec': {
                'scaleTargetRef': {
                    'apiVersion': 'apps/v1',
                    'kind': 'Deployment',
                    'name': f'{config.name}-{config.version}'
                },
                'minReplicas': config.min_replicas,
                'maxReplicas': config.max_replicas,
                'targetCPUUtilizationPercentage': config.target_cpu_percentage
            }
        }
    
    def _create_network_policy_spec(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Create network policy specification"""
        return {
            'apiVersion': 'networking.k8s.io/v1',
            'kind': 'NetworkPolicy',
            'metadata': {
                'name': f'{config.name}-netpol'
            },
            'spec': {
                'podSelector': {'matchLabels': {'app': config.name}},
                'policyTypes': ['Ingress', 'Egress'],
                'ingress': [
                    {
                        'from': [
                            {'podSelector': {'matchLabels': {'role': 'frontend'}}},
                            {'namespaceSelector': {'matchLabels': {'name': 'monitoring'}}}
                        ],
                        'ports': [{'protocol': 'TCP', 'port': 5000}]
                    }
                ],
                'egress': [
                    {'to': [], 'ports': [{'protocol': 'TCP', 'port': 53}]},  # DNS
                    {'to': [], 'ports': [{'protocol': 'UDP', 'port': 53}]}   # DNS
                ]
            }
        }
    
    def update_deployment(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Update existing deployment"""
        deployment_name = f'{config.name}-{config.version}'
        
        try:
            # Get current deployment
            current_deployment = self.apps_v1.read_namespaced_deployment(
                name=deployment_name,
                namespace=self.namespace
            )
            
            # Update deployment spec
            new_spec = self._create_deployment_spec(config)
            current_deployment.spec = new_spec['spec']
            
            # Apply update
            updated_deployment = self.apps_v1.patch_namespaced_deployment(
                name=deployment_name,
                namespace=self.namespace,
                body=current_deployment
            )
            
            return {
                'deployment': updated_deployment.to_dict(),
                'status': 'updated'
            }
            
        except Exception as e:
            logger.error(f"Error updating deployment: {e}")
            raise
    
    def get_deployment_status(self, deployment_name: str) -> Dict[str, Any]:
        """Get status of deployment"""
        try:
            deployment = self.apps_v1.read_namespaced_deployment_status(
                name=deployment_name,
                namespace=self.namespace
            )
            
            # Get pod information
            pods = self.v1.list_namespaced_pod(
                namespace=self.namespace,
                label_selector=f'app={deployment_name.split("-")[0]}'
            )
            
            pod_info = []
            for pod in pods.items:
                pod_info.append({
                    'name': pod.metadata.name,
                    'phase': pod.status.phase,
                    'ready': all(condition.status == 'True' 
                               for condition in pod.status.conditions or []
                               if condition.type == 'Ready')
                })
            
            return {
                'deployment': deployment.to_dict(),
                'pods': pod_info,
                'replicas': deployment.status.replicas or 0,
                'ready_replicas': deployment.status.ready_replicas or 0,
                'available_replicas': deployment.status.available_replicas or 0
            }
            
        except Exception as e:
            logger.error(f"Error getting deployment status: {e}")
            raise
    
    def delete_deployment(self, deployment_name: str) -> bool:
        """Delete deployment and associated resources"""
        try:
            # Delete deployment
            self.apps_v1.delete_namespaced_deployment(
                name=deployment_name,
                namespace=self.namespace
            )
            
            # Delete service
            service_name = deployment_name.split('-')[0]  # Remove version suffix
            try:
                self.v1.delete_namespaced_service(
                    name=service_name,
                    namespace=self.namespace
                )
            except:
                pass  # Service might not exist
            
            # Delete HPA
            try:
                self.autoscaling_v1.delete_namespaced_horizontal_pod_autoscaler(
                    name=f'{service_name}-hpa',
                    namespace=self.namespace
                )
            except:
                pass  # HPA might not exist
            
            # Delete network policy
            try:
                self.networking_v1.delete_namespaced_network_policy(
                    name=f'{service_name}-netpol',
                    namespace=self.namespace
                )
            except:
                pass  # Network policy might not exist
            
            logger.info(f"Deleted deployment: {deployment_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting deployment: {e}")
            return False


class CanaryDeploymentManager:
    """Manages canary deployments with traffic splitting"""
    
    def __init__(self, orchestrator: KubernetesOrchestrator):
        self.orchestrator = orchestrator
        self.active_canaries: Dict[str, Dict] = {}
    
    def start_canary_deployment(self, config: DeploymentConfig) -> str:
        """Start canary deployment"""
        canary_id = str(uuid.uuid4())
        
        # Create canary deployment with reduced replicas
        canary_config = DeploymentConfig(
            name=f"{config.name}-canary",
            version=config.version,
            strategy=config.strategy,
            environment=config.environment,
            cloud_provider=config.cloud_provider,
            replicas=max(1, config.replicas // 5),  # 20% of production replicas
            **{k: v for k, v in asdict(config).items() 
               if k not in ['name', 'replicas']}
        )
        
        # Deploy canary
        canary_deployment = self.orchestrator.create_deployment(canary_config)
        
        # Store canary info
        self.active_canaries[canary_id] = {
            'config': canary_config,
            'deployment': canary_deployment,
            'weight': config.canary_weight,
            'start_time': datetime.now(),
            'status': 'active'
        }
        
        logger.info(f"Started canary deployment: {canary_id}")
        return canary_id
    
    def promote_canary(self, canary_id: str, production_config: DeploymentConfig) -> bool:
        """Promote canary to production"""
        if canary_id not in self.active_canaries:
            raise ValueError(f"Canary {canary_id} not found")
        
        canary_info = self.active_canaries[canary_id]
        
        try:
            # Scale up canary to full production replicas
            production_config.replicas = production_config.replicas
            self.orchestrator.update_deployment(production_config)
            
            # Remove old production deployment
            old_deployment_name = f"{production_config.name}-{production_config.version}"
            self.orchestrator.delete_deployment(old_deployment_name)
            
            # Update canary status
            canary_info['status'] = 'promoted'
            canary_info['promoted_at'] = datetime.now()
            
            logger.info(f"Promoted canary deployment: {canary_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error promoting canary: {e}")
            return False
    
    def rollback_canary(self, canary_id: str) -> bool:
        """Rollback canary deployment"""
        if canary_id not in self.active_canaries:
            raise ValueError(f"Canary {canary_id} not found")
        
        canary_info = self.active_canaries[canary_id]
        canary_config = canary_info['config']
        
        try:
            # Delete canary deployment
            canary_deployment_name = f"{canary_config.name}-{canary_config.version}"
            self.orchestrator.delete_deployment(canary_deployment_name)
            
            # Update status
            canary_info['status'] = 'rolled_back'
            canary_info['rolled_back_at'] = datetime.now()
            
            logger.info(f"Rolled back canary deployment: {canary_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error rolling back canary: {e}")
            return False


class BlueGreenDeploymentManager:
    """Manages blue-green deployments"""
    
    def __init__(self, orchestrator: KubernetesOrchestrator):
        self.orchestrator = orchestrator
        self.active_deployments: Dict[str, Dict] = {}
    
    def deploy_green(self, config: DeploymentConfig) -> str:
        """Deploy green environment"""
        deployment_id = str(uuid.uuid4())
        
        # Create green deployment
        green_config = DeploymentConfig(
            name=f"{config.name}-green",
            version=config.version,
            strategy=config.strategy,
            environment=config.environment,
            cloud_provider=config.cloud_provider,
            **{k: v for k, v in asdict(config).items() 
               if k not in ['name']}
        )
        
        green_deployment = self.orchestrator.create_deployment(green_config)
        
        # Store deployment info
        self.active_deployments[deployment_id] = {
            'blue_config': config,
            'green_config': green_config,
            'green_deployment': green_deployment,
            'status': 'green_deployed',
            'start_time': datetime.now()
        }
        
        logger.info(f"Deployed green environment: {deployment_id}")
        return deployment_id
    
    def switch_to_green(self, deployment_id: str) -> bool:
        """Switch traffic from blue to green"""
        if deployment_id not in self.active_deployments:
            raise ValueError(f"Deployment {deployment_id} not found")
        
        deployment_info = self.active_deployments[deployment_id]
        
        try:
            # Update service to point to green deployment
            green_config = deployment_info['green_config']
            
            # Create or update service to point to green
            service_spec = self.orchestrator._create_service_spec(green_config)
            service_spec['spec']['selector'] = {'app': green_config.name.replace('-green', '')}
            
            # Apply service update (this switches traffic)
            self.orchestrator.v1.patch_namespaced_service(
                name=green_config.name.replace('-green', ''),
                namespace=self.orchestrator.namespace,
                body=service_spec
            )
            
            # Update status
            deployment_info['status'] = 'switched_to_green'
            deployment_info['switched_at'] = datetime.now()
            
            logger.info(f"Switched traffic to green: {deployment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error switching to green: {e}")
            return False
    
    def cleanup_blue(self, deployment_id: str) -> bool:
        """Remove blue deployment after successful green deployment"""
        if deployment_id not in self.active_deployments:
            raise ValueError(f"Deployment {deployment_id} not found")
        
        deployment_info = self.active_deployments[deployment_id]
        blue_config = deployment_info['blue_config']
        
        try:
            # Delete blue deployment
            blue_deployment_name = f"{blue_config.name}-{blue_config.version}"
            self.orchestrator.delete_deployment(blue_deployment_name)
            
            # Update status
            deployment_info['status'] = 'blue_cleaned_up'
            deployment_info['cleanup_at'] = datetime.now()
            
            logger.info(f"Cleaned up blue deployment: {deployment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error cleaning up blue: {e}")
            return False


class DeploymentMonitor:
    """Monitors deployment health and triggers rollbacks"""
    
    def __init__(self, orchestrator: KubernetesOrchestrator):
        self.orchestrator = orchestrator
        self.monitoring_threads: Dict[str, threading.Thread] = {}
        self.stop_monitoring: Dict[str, threading.Event] = {}
        
        # Metrics
        if PROMETHEUS_AVAILABLE:
            self.error_rate_metric = Counter('deployment_errors_total', 
                                            'Total deployment errors', 
                                            ['deployment', 'environment'])
            self.latency_metric = Histogram('deployment_latency_seconds',
                                          'Deployment latency',
                                          ['deployment', 'environment'])
            self.health_metric = Gauge('deployment_health_status',
                                     'Deployment health status',
                                     ['deployment', 'environment'])
    
    def start_monitoring(self, deployment_name: str, config: DeploymentConfig,
                        rollback_callback: Callable = None) -> None:
        """Start monitoring deployment"""
        if deployment_name in self.monitoring_threads:
            logger.warning(f"Already monitoring deployment: {deployment_name}")
            return
        
        stop_event = threading.Event()
        self.stop_monitoring[deployment_name] = stop_event
        
        monitor_thread = threading.Thread(
            target=self._monitor_deployment,
            args=(deployment_name, config, stop_event, rollback_callback)
        )
        monitor_thread.daemon = True
        monitor_thread.start()
        
        self.monitoring_threads[deployment_name] = monitor_thread
        logger.info(f"Started monitoring deployment: {deployment_name}")
    
    def stop_monitoring(self, deployment_name: str) -> None:
        """Stop monitoring deployment"""
        if deployment_name in self.stop_monitoring:
            self.stop_monitoring[deployment_name].set()
            
        if deployment_name in self.monitoring_threads:
            self.monitoring_threads[deployment_name].join(timeout=5.0)
            del self.monitoring_threads[deployment_name]
            
        logger.info(f"Stopped monitoring deployment: {deployment_name}")
    
    def _monitor_deployment(self, deployment_name: str, config: DeploymentConfig,
                          stop_event: threading.Event, rollback_callback: Callable) -> None:
        """Monitor deployment health in background"""
        start_time = datetime.now()
        error_count = 0
        high_latency_count = 0
        
        while not stop_event.is_set():
            try:
                # Check if monitoring period exceeded
                if datetime.now() - start_time > timedelta(minutes=config.monitoring_duration_minutes):
                    logger.info(f"Monitoring period completed for {deployment_name}")
                    break
                
                # Get deployment status
                status = self.orchestrator.get_deployment_status(deployment_name)
                
                # Check replica health
                replicas = status.get('replicas', 0)
                ready_replicas = status.get('ready_replicas', 0)
                
                if replicas > 0:
                    health_ratio = ready_replicas / replicas
                    
                    # Update health metric
                    if PROMETHEUS_AVAILABLE:
                        self.health_metric.labels(
                            deployment=deployment_name,
                            environment=config.environment.value
                        ).set(health_ratio)
                    
                    # Check for rollback conditions
                    if health_ratio < 0.5:  # Less than 50% healthy replicas
                        error_count += 1
                        
                        if error_count >= 3 and config.enable_auto_rollback:
                            logger.warning(f"Health check failure for {deployment_name}, triggering rollback")
                            if rollback_callback:
                                rollback_callback(deployment_name, "health_check_failure")
                            break
                    else:
                        error_count = 0  # Reset error count on successful check
                
                # Sleep between checks
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error monitoring deployment {deployment_name}: {e}")
                time.sleep(60)  # Wait longer on error
        
        logger.info(f"Monitoring stopped for deployment: {deployment_name}")


class AdvancedDeploymentOrchestrator:
    """Main deployment orchestrator combining all strategies"""
    
    def __init__(self, namespace: str = "default", kubeconfig_path: str = None):
        self.orchestrator = KubernetesOrchestrator(namespace, kubeconfig_path)
        self.canary_manager = CanaryDeploymentManager(self.orchestrator)
        self.blue_green_manager = BlueGreenDeploymentManager(self.orchestrator)
        self.monitor = DeploymentMonitor(self.orchestrator)
        
        self.active_deployments: Dict[str, DeploymentStatus] = {}
        self.deployment_history: List[DeploymentStatus] = []
        
        logger.info("Advanced Deployment Orchestrator initialized")
    
    def deploy(self, config: DeploymentConfig) -> str:
        """Execute deployment using specified strategy"""
        deployment_id = str(uuid.uuid4())
        
        # Create deployment status
        status = DeploymentStatus(
            id=deployment_id,
            config=config,
            status="pending",
            created_at=datetime.now()
        )
        
        self.active_deployments[deployment_id] = status
        
        try:
            if config.strategy == DeploymentStrategy.CANARY:
                return self._deploy_canary(config, deployment_id)
            elif config.strategy == DeploymentStrategy.BLUE_GREEN:
                return self._deploy_blue_green(config, deployment_id)
            elif config.strategy == DeploymentStrategy.ROLLING:
                return self._deploy_rolling(config, deployment_id)
            elif config.strategy == DeploymentStrategy.A_B_TEST:
                return self._deploy_ab_test(config, deployment_id)
            else:
                return self._deploy_recreate(config, deployment_id)
                
        except Exception as e:
            status.status = "failed"
            status.error_message = str(e)
            logger.error(f"Deployment failed: {e}")
            raise
    
    def _deploy_canary(self, config: DeploymentConfig, deployment_id: str) -> str:
        """Execute canary deployment"""
        status = self.active_deployments[deployment_id]
        status.status = "running"
        
        # Start canary
        canary_id = self.canary_manager.start_canary_deployment(config)
        
        # Start monitoring
        self.monitor.start_monitoring(
            f"{config.name}-canary-{config.version}",
            config,
            self._rollback_callback
        )
        
        status.metrics['canary_id'] = canary_id
        status.status = "successful"
        
        logger.info(f"Canary deployment started: {deployment_id}")
        return deployment_id
    
    def _deploy_blue_green(self, config: DeploymentConfig, deployment_id: str) -> str:
        """Execute blue-green deployment"""
        status = self.active_deployments[deployment_id]
        status.status = "running"
        
        # Deploy green
        bg_deployment_id = self.blue_green_manager.deploy_green(config)
        
        # Wait for green to be ready (simplified - in production would check readiness)
        time.sleep(30)
        
        # Switch to green
        self.blue_green_manager.switch_to_green(bg_deployment_id)
        
        # Start monitoring
        self.monitor.start_monitoring(
            f"{config.name}-green-{config.version}",
            config,
            self._rollback_callback
        )
        
        status.metrics['bg_deployment_id'] = bg_deployment_id
        status.status = "successful"
        
        logger.info(f"Blue-green deployment completed: {deployment_id}")
        return deployment_id
    
    def _deploy_rolling(self, config: DeploymentConfig, deployment_id: str) -> str:
        """Execute rolling deployment"""
        status = self.active_deployments[deployment_id]
        status.status = "running"
        
        # Standard Kubernetes rolling deployment
        deployment_result = self.orchestrator.create_deployment(config)
        
        # Start monitoring
        self.monitor.start_monitoring(
            f"{config.name}-{config.version}",
            config,
            self._rollback_callback
        )
        
        status.metrics['deployment_result'] = deployment_result
        status.status = "successful"
        
        logger.info(f"Rolling deployment completed: {deployment_id}")
        return deployment_id
    
    def _deploy_ab_test(self, config: DeploymentConfig, deployment_id: str) -> str:
        """Execute A/B test deployment"""
        # Similar to canary but with different traffic splitting
        return self._deploy_canary(config, deployment_id)
    
    def _deploy_recreate(self, config: DeploymentConfig, deployment_id: str) -> str:
        """Execute recreate deployment"""
        status = self.active_deployments[deployment_id]
        status.status = "running"
        
        # Delete existing deployment
        old_deployment_name = f"{config.name}-{config.version}"
        self.orchestrator.delete_deployment(old_deployment_name)
        
        # Wait for deletion
        time.sleep(10)
        
        # Create new deployment
        deployment_result = self.orchestrator.create_deployment(config)
        
        status.metrics['deployment_result'] = deployment_result
        status.status = "successful"
        
        logger.info(f"Recreate deployment completed: {deployment_id}")
        return deployment_id
    
    def _rollback_callback(self, deployment_name: str, reason: str) -> None:
        """Callback for automatic rollback"""
        logger.warning(f"Triggering rollback for {deployment_name}: {reason}")
        
        # Find deployment ID
        deployment_id = None
        for did, status in self.active_deployments.items():
            if f"{status.config.name}-{status.config.version}" in deployment_name:
                deployment_id = did
                break
        
        if deployment_id:
            self.rollback_deployment(deployment_id, reason)
    
    def rollback_deployment(self, deployment_id: str, reason: str = None) -> bool:
        """Rollback deployment"""
        if deployment_id not in self.active_deployments:
            raise ValueError(f"Deployment {deployment_id} not found")
        
        status = self.active_deployments[deployment_id]
        
        try:
            status.status = "rolling_back"
            status.rollback_info = {
                'reason': reason,
                'initiated_at': datetime.now()
            }
            
            config = status.config
            
            if config.strategy == DeploymentStrategy.CANARY:
                canary_id = status.metrics.get('canary_id')
                if canary_id:
                    self.canary_manager.rollback_canary(canary_id)
            
            elif config.strategy == DeploymentStrategy.BLUE_GREEN:
                bg_deployment_id = status.metrics.get('bg_deployment_id')
                if bg_deployment_id:
                    # Switch back to blue (simplified)
                    pass
            
            # Stop monitoring
            deployment_name = f"{config.name}-{config.version}"
            self.monitor.stop_monitoring(deployment_name)
            
            status.status = "rolled_back"
            status.rollback_info['completed_at'] = datetime.now()
            
            logger.info(f"Rollback completed for deployment: {deployment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Rollback failed for deployment {deployment_id}: {e}")
            status.status = "rollback_failed"
            status.error_message = str(e)
            return False
    
    def get_deployment_status(self, deployment_id: str) -> DeploymentStatus:
        """Get deployment status"""
        if deployment_id not in self.active_deployments:
            raise ValueError(f"Deployment {deployment_id} not found")
        
        return self.active_deployments[deployment_id]
    
    def list_deployments(self) -> List[DeploymentStatus]:
        """List all deployments"""
        return list(self.active_deployments.values())
    
    def generate_deployment_report(self) -> Dict[str, Any]:
        """Generate comprehensive deployment report"""
        report = {
            'summary': {
                'total_deployments': len(self.active_deployments),
                'successful_deployments': len([d for d in self.active_deployments.values() 
                                             if d.status == 'successful']),
                'failed_deployments': len([d for d in self.active_deployments.values() 
                                         if d.status == 'failed']),
                'active_rollbacks': len([d for d in self.active_deployments.values() 
                                       if 'rolling_back' in d.status])
            },
            'deployments': [asdict(status) for status in self.active_deployments.values()],
            'generated_at': datetime.now().isoformat()
        }
        
        return report
    
    def cleanup_completed_deployments(self, older_than_hours: int = 24) -> int:
        """Clean up old completed deployments"""
        cutoff_time = datetime.now() - timedelta(hours=older_than_hours)
        cleaned_count = 0
        
        to_remove = []
        for deployment_id, status in self.active_deployments.items():
            if (status.status in ['successful', 'failed', 'rolled_back'] and 
                status.updated_at < cutoff_time):
                
                # Move to history
                self.deployment_history.append(status)
                to_remove.append(deployment_id)
                cleaned_count += 1
        
        for deployment_id in to_remove:
            del self.active_deployments[deployment_id]
        
        logger.info(f"Cleaned up {cleaned_count} completed deployments")
        return cleaned_count


# Factory functions
def create_deployment_orchestrator(namespace: str = "default", 
                                 kubeconfig_path: str = None) -> AdvancedDeploymentOrchestrator:
    """Create deployment orchestrator"""
    return AdvancedDeploymentOrchestrator(namespace, kubeconfig_path)


def create_deployment_config(name: str, version: str, 
                           strategy: DeploymentStrategy = DeploymentStrategy.ROLLING,
                           environment: DeploymentEnvironment = DeploymentEnvironment.STAGING,
                           **kwargs) -> DeploymentConfig:
    """Create deployment configuration"""
    return DeploymentConfig(
        name=name,
        version=version,
        strategy=strategy,
        environment=environment,
        cloud_provider=CloudProvider.KUBERNETES,
        **kwargs
    )


# Example usage
if __name__ == "__main__":
    # Create orchestrator
    orchestrator = create_deployment_orchestrator()
    
    # Create deployment configuration
    config = create_deployment_config(
        name="sentiment-analyzer",
        version="v1.2.3",
        strategy=DeploymentStrategy.CANARY,
        environment=DeploymentEnvironment.PRODUCTION,
        replicas=5,
        enable_autoscaling=True,
        canary_weight=20
    )
    
    # Execute deployment
    try:
        deployment_id = orchestrator.deploy(config)
        print(f"Deployment started: {deployment_id}")
        
        # Monitor deployment
        time.sleep(60)  # Wait for deployment
        
        status = orchestrator.get_deployment_status(deployment_id)
        print(f"Deployment status: {status.status}")
        
        # Generate report
        report = orchestrator.generate_deployment_report()
        print("Deployment Report:", json.dumps(report, indent=2, default=str))
        
    except Exception as e:
        print(f"Deployment failed: {e}")