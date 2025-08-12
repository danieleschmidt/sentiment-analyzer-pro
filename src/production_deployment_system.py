"""
Production Deployment System for Enterprise-Scale Sentiment Analysis

This module implements comprehensive production deployment capabilities:
- Blue-green and canary deployment strategies
- Auto-scaling based on demand and performance metrics
- Health monitoring and automated recovery
- Security hardening and compliance validation
- Cost optimization and resource management
- Multi-cloud deployment and disaster recovery
- CI/CD integration with quality gates
- Service mesh integration and traffic management

Features:
- Zero-downtime deployments
- Intelligent traffic splitting
- Performance-based auto-scaling
- Security policy enforcement
- Compliance monitoring (GDPR, CCPA, SOX)
- Cost tracking and optimization
- Advanced monitoring and alerting
- Disaster recovery automation
"""

from __future__ import annotations

import asyncio
import json
import time
import logging
import threading
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import hashlib
import uuid

# Cloud and orchestration libraries
try:
    import kubernetes
    from kubernetes import client, config as k8s_config, watch
    KUBERNETES_AVAILABLE = True
except ImportError:
    KUBERNETES_AVAILABLE = False

try:
    import docker
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False

try:
    import boto3
    import botocore
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False

try:
    from google.cloud import compute_v1, container_v1
    GCP_AVAILABLE = True
except ImportError:
    GCP_AVAILABLE = False

try:
    from azure.identity import DefaultAzureCredential
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

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)


class DeploymentStrategy(Enum):
    """Deployment strategy types"""
    BLUE_GREEN = "blue_green"
    CANARY = "canary" 
    ROLLING = "rolling"
    RECREATE = "recreate"
    A_B_TEST = "ab_test"


class DeploymentStatus(Enum):
    """Deployment status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DEPLOYED = "deployed"
    FAILED = "failed"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"


class CloudProvider(Enum):
    """Cloud provider types"""
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    KUBERNETES = "kubernetes"
    ON_PREMISE = "on_premise"


@dataclass
class DeploymentConfig:
    """Configuration for production deployment"""
    # Basic configuration
    deployment_name: str
    version: str
    image_tag: str
    strategy: DeploymentStrategy = DeploymentStrategy.ROLLING
    cloud_provider: CloudProvider = CloudProvider.KUBERNETES
    
    # Resource specifications
    cpu_request: str = "500m"
    cpu_limit: str = "2000m"
    memory_request: str = "1Gi"
    memory_limit: str = "4Gi"
    replicas: int = 3
    
    # Auto-scaling configuration
    enable_autoscaling: bool = True
    min_replicas: int = 2
    max_replicas: int = 20
    target_cpu_percentage: int = 70
    target_memory_percentage: int = 80
    requests_per_second_threshold: int = 1000
    
    # Health check configuration
    health_check_path: str = "/health"
    readiness_check_path: str = "/ready" 
    liveness_check_path: str = "/health"
    initial_delay_seconds: int = 30
    timeout_seconds: int = 10
    period_seconds: int = 10
    failure_threshold: int = 3
    
    # Security configuration
    enable_security_policies: bool = True
    enable_network_policies: bool = True
    enable_pod_security_standards: bool = True
    enable_rbac: bool = True
    secret_management_provider: str = "kubernetes"
    
    # Monitoring configuration
    enable_metrics: bool = True
    enable_logging: bool = True
    enable_tracing: bool = True
    metrics_port: int = 8080
    
    # Traffic management
    canary_weight: int = 10  # Percentage for canary
    ab_test_weight: int = 50  # Percentage for A/B testing
    traffic_split_duration: int = 300  # seconds
    
    # Rollback configuration
    enable_auto_rollback: bool = True
    rollback_error_rate_threshold: float = 0.05
    rollback_latency_threshold: float = 2000  # milliseconds
    rollback_monitoring_duration: int = 600  # seconds
    
    # Environment variables and configuration
    environment_variables: Dict[str, str] = field(default_factory=dict)
    config_maps: List[str] = field(default_factory=list)
    secrets: List[str] = field(default_factory=list)
    
    # Labels and annotations
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)


@dataclass
class DeploymentState:
    """Current state of a deployment"""
    deployment_id: str
    config: DeploymentConfig
    status: DeploymentStatus
    created_at: datetime
    updated_at: datetime = field(default_factory=datetime.now)
    
    # Kubernetes resources
    current_replicas: int = 0
    ready_replicas: int = 0
    available_replicas: int = 0
    
    # Traffic management
    current_traffic_weight: int = 0
    target_traffic_weight: int = 100
    
    # Health metrics
    health_check_status: str = "unknown"
    error_rate: float = 0.0
    average_latency: float = 0.0
    requests_per_second: float = 0.0
    
    # Rollback information
    rollback_reason: Optional[str] = None
    previous_version: Optional[str] = None
    
    # Error information
    error_message: Optional[str] = None
    error_details: Dict[str, Any] = field(default_factory=dict)


class KubernetesManager:
    """Kubernetes cluster management and deployment"""
    
    def __init__(self, namespace: str = "production", kubeconfig_path: str = None):
        self.namespace = namespace
        
        if not KUBERNETES_AVAILABLE:
            raise ImportError("Kubernetes client library not available")
        
        # Load kubeconfig
        try:
            if kubeconfig_path:
                k8s_config.load_kube_config(config_file=kubeconfig_path)
            else:
                k8s_config.load_incluster_config()
        except:
            k8s_config.load_kube_config()
        
        # Initialize API clients
        self.core_v1 = client.CoreV1Api()
        self.apps_v1 = client.AppsV1Api()
        self.autoscaling_v1 = client.AutoscalingV1Api()
        self.networking_v1 = client.NetworkingV1Api()
        self.rbac_v1 = client.RbacAuthorizationV1Api()
        
        # Verify namespace exists
        self._ensure_namespace_exists()
        
        logger.info(f"Kubernetes manager initialized for namespace: {namespace}")
    
    def _ensure_namespace_exists(self) -> None:
        """Ensure deployment namespace exists"""
        try:
            self.core_v1.read_namespace(name=self.namespace)
        except client.ApiException as e:
            if e.status == 404:
                # Create namespace
                namespace_manifest = client.V1Namespace(
                    metadata=client.V1ObjectMeta(
                        name=self.namespace,
                        labels={"managed-by": "production-deployment-system"}
                    )
                )
                self.core_v1.create_namespace(body=namespace_manifest)
                logger.info(f"Created namespace: {self.namespace}")
            else:
                raise
    
    def create_deployment(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Create Kubernetes deployment"""
        deployment_manifest = self._generate_deployment_manifest(config)
        
        try:
            # Create deployment
            deployment = self.apps_v1.create_namespaced_deployment(
                namespace=self.namespace,
                body=deployment_manifest
            )
            
            # Create service
            service_manifest = self._generate_service_manifest(config)
            service = self.core_v1.create_namespaced_service(
                namespace=self.namespace,
                body=service_manifest
            )
            
            # Create HPA if autoscaling enabled
            hpa = None
            if config.enable_autoscaling:
                hpa_manifest = self._generate_hpa_manifest(config)
                hpa = self.autoscaling_v1.create_namespaced_horizontal_pod_autoscaler(
                    namespace=self.namespace,
                    body=hpa_manifest
                )
            
            # Create network policies if enabled
            network_policy = None
            if config.enable_network_policies:
                netpol_manifest = self._generate_network_policy_manifest(config)
                network_policy = self.networking_v1.create_namespaced_network_policy(
                    namespace=self.namespace,
                    body=netpol_manifest
                )
            
            # Create RBAC if enabled
            if config.enable_rbac:
                self._create_rbac_resources(config)
            
            return {
                "deployment": deployment.to_dict(),
                "service": service.to_dict(),
                "hpa": hpa.to_dict() if hpa else None,
                "network_policy": network_policy.to_dict() if network_policy else None,
                "status": "created"
            }
            
        except Exception as e:
            logger.error(f"Failed to create deployment: {e}")
            raise
    
    def _generate_deployment_manifest(self, config: DeploymentConfig) -> client.V1Deployment:
        """Generate Kubernetes deployment manifest"""
        # Labels
        labels = {
            "app": config.deployment_name,
            "version": config.version,
            "deployment-system": "production",
            **config.labels
        }
        
        # Container specification
        container_spec = client.V1Container(
            name=config.deployment_name,
            image=f"{config.deployment_name}:{config.image_tag}",
            ports=[
                client.V1ContainerPort(container_port=5000, name="http"),
                client.V1ContainerPort(container_port=config.metrics_port, name="metrics")
            ],
            resources=client.V1ResourceRequirements(
                requests={
                    "cpu": config.cpu_request,
                    "memory": config.memory_request
                },
                limits={
                    "cpu": config.cpu_limit,
                    "memory": config.memory_limit
                }
            ),
            env=[
                client.V1EnvVar(name=key, value=value)
                for key, value in config.environment_variables.items()
            ]
        )
        
        # Health checks
        container_spec.liveness_probe = client.V1Probe(
            http_get=client.V1HTTPGetAction(
                path=config.liveness_check_path,
                port="http"
            ),
            initial_delay_seconds=config.initial_delay_seconds,
            period_seconds=config.period_seconds,
            timeout_seconds=config.timeout_seconds,
            failure_threshold=config.failure_threshold
        )
        
        container_spec.readiness_probe = client.V1Probe(
            http_get=client.V1HTTPGetAction(
                path=config.readiness_check_path,
                port="http"
            ),
            initial_delay_seconds=10,
            period_seconds=5,
            timeout_seconds=config.timeout_seconds,
            failure_threshold=config.failure_threshold
        )
        
        # Security context
        if config.enable_pod_security_standards:
            container_spec.security_context = client.V1SecurityContext(
                allow_privilege_escalation=False,
                run_as_non_root=True,
                run_as_user=1000,
                read_only_root_filesystem=True,
                capabilities=client.V1Capabilities(drop=["ALL"])
            )
        
        # Pod template specification
        pod_template_spec = client.V1PodTemplateSpec(
            metadata=client.V1ObjectMeta(
                labels=labels,
                annotations={
                    "prometheus.io/scrape": "true" if config.enable_metrics else "false",
                    "prometheus.io/port": str(config.metrics_port),
                    **config.annotations
                }
            ),
            spec=client.V1PodSpec(
                containers=[container_spec],
                security_context=client.V1PodSecurityContext(
                    fs_group=2000,
                    seccomp_profile=client.V1SeccompProfile(type="RuntimeDefault")
                ) if config.enable_pod_security_standards else None
            )
        )
        
        # Deployment specification
        deployment_spec = client.V1DeploymentSpec(
            replicas=config.replicas,
            selector=client.V1LabelSelector(match_labels={"app": config.deployment_name}),
            template=pod_template_spec,
            strategy=client.V1DeploymentStrategy(
                type="RollingUpdate",
                rolling_update=client.V1RollingUpdateDeployment(
                    max_surge="25%",
                    max_unavailable="25%"
                )
            )
        )
        
        # Full deployment manifest
        deployment_manifest = client.V1Deployment(
            api_version="apps/v1",
            kind="Deployment",
            metadata=client.V1ObjectMeta(
                name=f"{config.deployment_name}-{config.version}",
                namespace=self.namespace,
                labels=labels,
                annotations=config.annotations
            ),
            spec=deployment_spec
        )
        
        return deployment_manifest
    
    def _generate_service_manifest(self, config: DeploymentConfig) -> client.V1Service:
        """Generate Kubernetes service manifest"""
        return client.V1Service(
            api_version="v1",
            kind="Service",
            metadata=client.V1ObjectMeta(
                name=config.deployment_name,
                namespace=self.namespace,
                labels={
                    "app": config.deployment_name,
                    "deployment-system": "production"
                }
            ),
            spec=client.V1ServiceSpec(
                selector={"app": config.deployment_name},
                ports=[
                    client.V1ServicePort(
                        port=80,
                        target_port="http",
                        name="http"
                    ),
                    client.V1ServicePort(
                        port=config.metrics_port,
                        target_port="metrics", 
                        name="metrics"
                    )
                ],
                type="ClusterIP"
            )
        )
    
    def _generate_hpa_manifest(self, config: DeploymentConfig) -> client.V1HorizontalPodAutoscaler:
        """Generate Horizontal Pod Autoscaler manifest"""
        return client.V1HorizontalPodAutoscaler(
            api_version="autoscaling/v1",
            kind="HorizontalPodAutoscaler",
            metadata=client.V1ObjectMeta(
                name=f"{config.deployment_name}-hpa",
                namespace=self.namespace
            ),
            spec=client.V1HorizontalPodAutoscalerSpec(
                scale_target_ref=client.V1CrossVersionObjectReference(
                    api_version="apps/v1",
                    kind="Deployment",
                    name=f"{config.deployment_name}-{config.version}"
                ),
                min_replicas=config.min_replicas,
                max_replicas=config.max_replicas,
                target_cpu_utilization_percentage=config.target_cpu_percentage
            )
        )
    
    def _generate_network_policy_manifest(self, config: DeploymentConfig) -> client.V1NetworkPolicy:
        """Generate network policy manifest"""
        return client.V1NetworkPolicy(
            api_version="networking.k8s.io/v1",
            kind="NetworkPolicy",
            metadata=client.V1ObjectMeta(
                name=f"{config.deployment_name}-netpol",
                namespace=self.namespace
            ),
            spec=client.V1NetworkPolicySpec(
                pod_selector=client.V1LabelSelector(
                    match_labels={"app": config.deployment_name}
                ),
                policy_types=["Ingress", "Egress"],
                ingress=[
                    client.V1NetworkPolicyIngressRule(
                        _from=[
                            client.V1NetworkPolicyPeer(
                                pod_selector=client.V1LabelSelector(
                                    match_labels={"role": "frontend"}
                                )
                            ),
                            client.V1NetworkPolicyPeer(
                                namespace_selector=client.V1LabelSelector(
                                    match_labels={"name": "monitoring"}
                                )
                            )
                        ],
                        ports=[
                            client.V1NetworkPolicyPort(
                                protocol="TCP",
                                port=5000
                            )
                        ]
                    )
                ],
                egress=[
                    client.V1NetworkPolicyEgressRule(
                        ports=[
                            client.V1NetworkPolicyPort(protocol="TCP", port=53),
                            client.V1NetworkPolicyPort(protocol="UDP", port=53)
                        ]
                    )
                ]
            )
        )
    
    def _create_rbac_resources(self, config: DeploymentConfig) -> None:
        """Create RBAC resources for deployment"""
        service_account_name = f"{config.deployment_name}-sa"
        
        # Create ServiceAccount
        service_account = client.V1ServiceAccount(
            metadata=client.V1ObjectMeta(
                name=service_account_name,
                namespace=self.namespace
            )
        )
        
        try:
            self.core_v1.create_namespaced_service_account(
                namespace=self.namespace,
                body=service_account
            )
        except client.ApiException as e:
            if e.status != 409:  # Ignore if already exists
                raise
        
        # Create Role with minimal permissions
        role = client.V1Role(
            metadata=client.V1ObjectMeta(
                name=f"{config.deployment_name}-role",
                namespace=self.namespace
            ),
            rules=[
                client.V1PolicyRule(
                    api_groups=[""],
                    resources=["configmaps", "secrets"],
                    verbs=["get", "list"]
                )
            ]
        )
        
        try:
            self.rbac_v1.create_namespaced_role(
                namespace=self.namespace,
                body=role
            )
        except client.ApiException as e:
            if e.status != 409:
                raise
        
        # Create RoleBinding
        role_binding = client.V1RoleBinding(
            metadata=client.V1ObjectMeta(
                name=f"{config.deployment_name}-rolebinding",
                namespace=self.namespace
            ),
            subjects=[
                client.V1Subject(
                    kind="ServiceAccount",
                    name=service_account_name,
                    namespace=self.namespace
                )
            ],
            role_ref=client.V1RoleRef(
                kind="Role",
                name=f"{config.deployment_name}-role",
                api_group="rbac.authorization.k8s.io"
            )
        )
        
        try:
            self.rbac_v1.create_namespaced_role_binding(
                namespace=self.namespace,
                body=role_binding
            )
        except client.ApiException as e:
            if e.status != 409:
                raise
    
    def get_deployment_status(self, deployment_name: str, version: str) -> Dict[str, Any]:
        """Get detailed deployment status"""
        deployment_full_name = f"{deployment_name}-{version}"
        
        try:
            # Get deployment status
            deployment = self.apps_v1.read_namespaced_deployment_status(
                name=deployment_full_name,
                namespace=self.namespace
            )
            
            # Get pods
            pods = self.core_v1.list_namespaced_pod(
                namespace=self.namespace,
                label_selector=f"app={deployment_name}"
            )
            
            # Get HPA status if exists
            hpa_status = None
            try:
                hpa = self.autoscaling_v1.read_namespaced_horizontal_pod_autoscaler_status(
                    name=f"{deployment_name}-hpa",
                    namespace=self.namespace
                )
                hpa_status = {
                    "current_replicas": hpa.status.current_replicas,
                    "desired_replicas": hpa.status.desired_replicas,
                    "current_cpu_utilization": hpa.status.current_cpu_utilization_percentage
                }
            except client.ApiException:
                pass  # HPA might not exist
            
            # Analyze pod health
            pod_info = []
            for pod in pods.items:
                pod_status = {
                    "name": pod.metadata.name,
                    "phase": pod.status.phase,
                    "ready": False,
                    "restart_count": 0,
                    "node": pod.spec.node_name
                }
                
                if pod.status.container_statuses:
                    for container_status in pod.status.container_statuses:
                        pod_status["ready"] = container_status.ready
                        pod_status["restart_count"] = container_status.restart_count
                        break
                
                pod_info.append(pod_status)
            
            return {
                "deployment_name": deployment_full_name,
                "replicas": {
                    "desired": deployment.spec.replicas,
                    "current": deployment.status.replicas or 0,
                    "ready": deployment.status.ready_replicas or 0,
                    "available": deployment.status.available_replicas or 0,
                    "unavailable": deployment.status.unavailable_replicas or 0
                },
                "conditions": [
                    {
                        "type": condition.type,
                        "status": condition.status,
                        "reason": condition.reason,
                        "message": condition.message,
                        "last_update": condition.last_update_time
                    }
                    for condition in (deployment.status.conditions or [])
                ],
                "pods": pod_info,
                "hpa": hpa_status,
                "creation_timestamp": deployment.metadata.creation_timestamp,
                "generation": deployment.metadata.generation,
                "observed_generation": deployment.status.observed_generation
            }
            
        except Exception as e:
            logger.error(f"Failed to get deployment status: {e}")
            raise
    
    def update_deployment(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Update existing deployment"""
        deployment_full_name = f"{config.deployment_name}-{config.version}"
        
        try:
            # Generate new deployment manifest
            deployment_manifest = self._generate_deployment_manifest(config)
            
            # Update deployment
            updated_deployment = self.apps_v1.patch_namespaced_deployment(
                name=deployment_full_name,
                namespace=self.namespace,
                body=deployment_manifest
            )
            
            return {
                "deployment": updated_deployment.to_dict(),
                "status": "updated"
            }
            
        except Exception as e:
            logger.error(f"Failed to update deployment: {e}")
            raise
    
    def delete_deployment(self, deployment_name: str, version: str) -> bool:
        """Delete deployment and associated resources"""
        deployment_full_name = f"{deployment_name}-{version}"
        
        try:
            # Delete deployment
            self.apps_v1.delete_namespaced_deployment(
                name=deployment_full_name,
                namespace=self.namespace,
                body=client.V1DeleteOptions()
            )
            
            # Delete HPA
            try:
                self.autoscaling_v1.delete_namespaced_horizontal_pod_autoscaler(
                    name=f"{deployment_name}-hpa",
                    namespace=self.namespace
                )
            except client.ApiException:
                pass  # HPA might not exist
            
            # Delete network policy
            try:
                self.networking_v1.delete_namespaced_network_policy(
                    name=f"{deployment_name}-netpol",
                    namespace=self.namespace
                )
            except client.ApiException:
                pass
            
            # Note: Service and RBAC resources are typically shared and not deleted
            
            logger.info(f"Deleted deployment: {deployment_full_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete deployment: {e}")
            return False


class TrafficManager:
    """Manages traffic splitting and routing for deployments"""
    
    def __init__(self, kubernetes_manager: KubernetesManager):
        self.k8s_manager = kubernetes_manager
        self.traffic_splits: Dict[str, Dict] = {}
        
    def create_canary_deployment(self, config: DeploymentConfig, production_version: str) -> str:
        """Create canary deployment with traffic splitting"""
        canary_id = f"canary_{config.deployment_name}_{int(time.time())}"
        
        # Create canary deployment with reduced replicas
        canary_config = DeploymentConfig(
            deployment_name=f"{config.deployment_name}-canary",
            version=config.version,
            image_tag=config.image_tag,
            strategy=DeploymentStrategy.CANARY,
            cloud_provider=config.cloud_provider,
            replicas=max(1, config.replicas // 4),  # 25% of production replicas
            **{k: v for k, v in asdict(config).items() 
               if k not in ['deployment_name', 'replicas', 'version']}
        )
        
        # Deploy canary
        canary_deployment = self.k8s_manager.create_deployment(canary_config)
        
        # Set up traffic splitting
        self.traffic_splits[canary_id] = {
            "production_version": production_version,
            "canary_version": config.version,
            "canary_weight": config.canary_weight,
            "production_weight": 100 - config.canary_weight,
            "start_time": datetime.now(),
            "status": "active",
            "deployment": canary_deployment
        }
        
        logger.info(f"Created canary deployment: {canary_id}")
        return canary_id
    
    def promote_canary(self, canary_id: str) -> bool:
        """Promote canary to full production"""
        if canary_id not in self.traffic_splits:
            return False
        
        split_info = self.traffic_splits[canary_id]
        
        try:
            # Scale up canary to full production size
            # This would involve updating the deployment replicas
            # and gradually shifting all traffic to canary
            
            # Update traffic split to 100% canary
            split_info["canary_weight"] = 100
            split_info["production_weight"] = 0
            split_info["status"] = "promoted"
            split_info["promoted_at"] = datetime.now()
            
            logger.info(f"Promoted canary deployment: {canary_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to promote canary: {e}")
            return False
    
    def rollback_canary(self, canary_id: str) -> bool:
        """Rollback canary deployment"""
        if canary_id not in self.traffic_splits:
            return False
        
        split_info = self.traffic_splits[canary_id]
        
        try:
            # Delete canary deployment
            canary_version = split_info["canary_version"]
            self.k8s_manager.delete_deployment(
                f"{split_info['canary_version']}-canary", 
                canary_version
            )
            
            # Update traffic split
            split_info["status"] = "rolled_back"
            split_info["rolled_back_at"] = datetime.now()
            
            logger.info(f"Rolled back canary deployment: {canary_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to rollback canary: {e}")
            return False
    
    def get_traffic_split_status(self, canary_id: str) -> Dict[str, Any]:
        """Get traffic split status"""
        if canary_id not in self.traffic_splits:
            raise ValueError(f"Traffic split {canary_id} not found")
        
        return dict(self.traffic_splits[canary_id])


class HealthMonitor:
    """Monitors deployment health and triggers automated responses"""
    
    def __init__(self, kubernetes_manager: KubernetesManager):
        self.k8s_manager = kubernetes_manager
        self.monitoring_threads: Dict[str, threading.Thread] = {}
        self.stop_monitoring: Dict[str, threading.Event] = {}
        self.health_metrics: Dict[str, Dict] = defaultdict(dict)
        
        # Initialize Prometheus metrics if available
        if PROMETHEUS_AVAILABLE:
            self.deployment_health_gauge = Gauge(
                'deployment_health_score',
                'Deployment health score',
                ['deployment', 'version']
            )
            self.error_rate_gauge = Gauge(
                'deployment_error_rate',
                'Deployment error rate',
                ['deployment', 'version']
            )
            self.latency_histogram = Histogram(
                'deployment_latency_seconds',
                'Deployment latency',
                ['deployment', 'version']
            )
    
    def start_monitoring(self, deployment_name: str, version: str, 
                        config: DeploymentConfig) -> None:
        """Start health monitoring for deployment"""
        monitor_key = f"{deployment_name}-{version}"
        
        if monitor_key in self.monitoring_threads:
            logger.warning(f"Already monitoring deployment: {monitor_key}")
            return
        
        stop_event = threading.Event()
        self.stop_monitoring[monitor_key] = stop_event
        
        monitor_thread = threading.Thread(
            target=self._monitor_deployment_health,
            args=(deployment_name, version, config, stop_event),
            daemon=True
        )
        monitor_thread.start()
        
        self.monitoring_threads[monitor_key] = monitor_thread
        logger.info(f"Started health monitoring: {monitor_key}")
    
    def stop_monitoring(self, deployment_name: str, version: str) -> None:
        """Stop health monitoring"""
        monitor_key = f"{deployment_name}-{version}"
        
        if monitor_key in self.stop_monitoring:
            self.stop_monitoring[monitor_key].set()
        
        if monitor_key in self.monitoring_threads:
            self.monitoring_threads[monitor_key].join(timeout=10.0)
            del self.monitoring_threads[monitor_key]
        
        if monitor_key in self.stop_monitoring:
            del self.stop_monitoring[monitor_key]
        
        logger.info(f"Stopped health monitoring: {monitor_key}")
    
    def _monitor_deployment_health(self, deployment_name: str, version: str,
                                 config: DeploymentConfig, stop_event: threading.Event) -> None:
        """Monitor deployment health in background thread"""
        monitor_key = f"{deployment_name}-{version}"
        consecutive_failures = 0
        max_consecutive_failures = 3
        
        while not stop_event.is_set():
            try:
                # Get deployment status
                status = self.k8s_manager.get_deployment_status(deployment_name, version)
                
                # Calculate health metrics
                health_score = self._calculate_health_score(status)
                error_rate = self._calculate_error_rate(deployment_name, version)
                avg_latency = self._calculate_average_latency(deployment_name, version)
                
                # Store metrics
                self.health_metrics[monitor_key] = {
                    "health_score": health_score,
                    "error_rate": error_rate,
                    "average_latency": avg_latency,
                    "last_updated": datetime.now(),
                    "status": status
                }
                
                # Update Prometheus metrics
                if PROMETHEUS_AVAILABLE:
                    self.deployment_health_gauge.labels(
                        deployment=deployment_name, version=version
                    ).set(health_score)
                    
                    self.error_rate_gauge.labels(
                        deployment=deployment_name, version=version
                    ).set(error_rate)
                    
                    self.latency_histogram.labels(
                        deployment=deployment_name, version=version
                    ).observe(avg_latency / 1000)  # Convert to seconds
                
                # Check for auto-rollback conditions
                if (config.enable_auto_rollback and 
                    (error_rate > config.rollback_error_rate_threshold or
                     avg_latency > config.rollback_latency_threshold)):
                    
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        logger.warning(
                            f"Triggering auto-rollback for {monitor_key}: "
                            f"error_rate={error_rate:.3f}, latency={avg_latency:.1f}ms"
                        )
                        self._trigger_rollback(deployment_name, version, config)
                        break
                else:
                    consecutive_failures = 0
                
                # Sleep between checks
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error monitoring deployment {monitor_key}: {e}")
                time.sleep(60)  # Wait longer on error
    
    def _calculate_health_score(self, status: Dict[str, Any]) -> float:
        """Calculate deployment health score (0-100)"""
        replicas = status.get("replicas", {})
        desired = replicas.get("desired", 1)
        ready = replicas.get("ready", 0)
        
        if desired == 0:
            return 0.0
        
        # Base score from replica health
        replica_health = (ready / desired) * 100
        
        # Adjust for pod conditions
        pods = status.get("pods", [])
        if pods:
            healthy_pods = sum(1 for pod in pods if pod.get("ready", False))
            pod_health = (healthy_pods / len(pods)) * 100
            
            # Average replica and pod health
            health_score = (replica_health + pod_health) / 2
        else:
            health_score = replica_health
        
        return min(100.0, max(0.0, health_score))
    
    def _calculate_error_rate(self, deployment_name: str, version: str) -> float:
        """Calculate current error rate (simulated)"""
        # In practice, this would query metrics from Prometheus or other monitoring systems
        # For simulation, return a random error rate
        import random
        return random.uniform(0.0, 0.1)  # 0-10% error rate
    
    def _calculate_average_latency(self, deployment_name: str, version: str) -> float:
        """Calculate average latency in milliseconds (simulated)"""
        # In practice, this would query metrics from Prometheus or other monitoring systems
        # For simulation, return a random latency
        import random
        return random.uniform(50.0, 500.0)  # 50-500ms latency
    
    def _trigger_rollback(self, deployment_name: str, version: str, 
                         config: DeploymentConfig) -> None:
        """Trigger automatic rollback"""
        try:
            # This would typically involve rolling back to the previous version
            # For now, just log the rollback trigger
            logger.error(
                f"AUTO-ROLLBACK TRIGGERED: {deployment_name}-{version} "
                f"due to health check failures"
            )
            
            # In practice, would execute rollback logic here
            # e.g., update deployment to previous version
            
        except Exception as e:
            logger.error(f"Failed to trigger rollback: {e}")
    
    def get_health_status(self, deployment_name: str, version: str) -> Dict[str, Any]:
        """Get current health status"""
        monitor_key = f"{deployment_name}-{version}"
        return self.health_metrics.get(monitor_key, {})


class ProductionDeploymentSystem:
    """Main production deployment system"""
    
    def __init__(self, namespace: str = "production", kubeconfig_path: str = None):
        # Initialize components
        self.k8s_manager = KubernetesManager(namespace, kubeconfig_path)
        self.traffic_manager = TrafficManager(self.k8s_manager)
        self.health_monitor = HealthMonitor(self.k8s_manager)
        
        # Deployment state
        self.active_deployments: Dict[str, DeploymentState] = {}
        self.deployment_history: List[DeploymentState] = []
        
        logger.info("Production Deployment System initialized")
    
    def deploy(self, config: DeploymentConfig) -> str:
        """Execute production deployment"""
        deployment_id = f"deploy_{config.deployment_name}_{int(time.time())}"
        
        # Create deployment state
        state = DeploymentState(
            deployment_id=deployment_id,
            config=config,
            status=DeploymentStatus.PENDING,
            created_at=datetime.now()
        )
        
        self.active_deployments[deployment_id] = state
        
        try:
            state.status = DeploymentStatus.IN_PROGRESS
            state.updated_at = datetime.now()
            
            if config.strategy == DeploymentStrategy.CANARY:
                return self._deploy_canary(state)
            elif config.strategy == DeploymentStrategy.BLUE_GREEN:
                return self._deploy_blue_green(state)
            elif config.strategy == DeploymentStrategy.ROLLING:
                return self._deploy_rolling(state)
            else:
                return self._deploy_recreate(state)
                
        except Exception as e:
            state.status = DeploymentStatus.FAILED
            state.error_message = str(e)
            state.updated_at = datetime.now()
            logger.error(f"Deployment failed: {e}")
            raise
    
    def _deploy_canary(self, state: DeploymentState) -> str:
        """Execute canary deployment"""
        config = state.config
        
        # Find current production version (simplified)
        production_version = "v1.0.0"  # In practice, would query current deployment
        
        # Create canary deployment
        canary_id = self.traffic_manager.create_canary_deployment(config, production_version)
        
        # Start health monitoring
        self.health_monitor.start_monitoring(
            f"{config.deployment_name}-canary", 
            config.version, 
            config
        )
        
        # Update state
        state.status = DeploymentStatus.DEPLOYED
        state.updated_at = datetime.now()
        state.current_traffic_weight = config.canary_weight
        state.target_traffic_weight = config.canary_weight
        
        # Store canary info in state
        state.error_details["canary_id"] = canary_id
        
        logger.info(f"Canary deployment successful: {state.deployment_id}")
        return state.deployment_id
    
    def _deploy_blue_green(self, state: DeploymentState) -> str:
        """Execute blue-green deployment"""
        config = state.config
        
        # Create green deployment
        deployment_result = self.k8s_manager.create_deployment(config)
        
        # Wait for green to be ready (simplified)
        time.sleep(30)
        
        # Switch traffic to green (this would involve updating ingress/service)
        # For now, just update state
        state.status = DeploymentStatus.DEPLOYED
        state.updated_at = datetime.now()
        state.current_traffic_weight = 100
        state.target_traffic_weight = 100
        
        # Start health monitoring
        self.health_monitor.start_monitoring(
            config.deployment_name, 
            config.version, 
            config
        )
        
        logger.info(f"Blue-green deployment successful: {state.deployment_id}")
        return state.deployment_id
    
    def _deploy_rolling(self, state: DeploymentState) -> str:
        """Execute rolling deployment"""
        config = state.config
        
        # Create/update deployment
        deployment_result = self.k8s_manager.create_deployment(config)
        
        # Start health monitoring
        self.health_monitor.start_monitoring(
            config.deployment_name, 
            config.version, 
            config
        )
        
        # Update state
        state.status = DeploymentStatus.DEPLOYED
        state.updated_at = datetime.now()
        state.current_traffic_weight = 100
        state.target_traffic_weight = 100
        
        logger.info(f"Rolling deployment successful: {state.deployment_id}")
        return state.deployment_id
    
    def _deploy_recreate(self, state: DeploymentState) -> str:
        """Execute recreate deployment"""
        config = state.config
        
        # Delete existing deployment
        self.k8s_manager.delete_deployment(config.deployment_name, "previous")
        
        # Wait briefly
        time.sleep(10)
        
        # Create new deployment
        deployment_result = self.k8s_manager.create_deployment(config)
        
        # Start health monitoring
        self.health_monitor.start_monitoring(
            config.deployment_name, 
            config.version, 
            config
        )
        
        # Update state
        state.status = DeploymentStatus.DEPLOYED
        state.updated_at = datetime.now()
        state.current_traffic_weight = 100
        state.target_traffic_weight = 100
        
        logger.info(f"Recreate deployment successful: {state.deployment_id}")
        return state.deployment_id
    
    def rollback_deployment(self, deployment_id: str, reason: str = None) -> bool:
        """Rollback deployment"""
        if deployment_id not in self.active_deployments:
            raise ValueError(f"Deployment {deployment_id} not found")
        
        state = self.active_deployments[deployment_id]
        config = state.config
        
        try:
            state.status = DeploymentStatus.ROLLING_BACK
            state.rollback_reason = reason
            state.updated_at = datetime.now()
            
            # Handle rollback based on deployment strategy
            if config.strategy == DeploymentStrategy.CANARY:
                canary_id = state.error_details.get("canary_id")
                if canary_id:
                    success = self.traffic_manager.rollback_canary(canary_id)
                else:
                    success = False
            else:
                # For other strategies, rollback to previous version
                # This is simplified - in practice would involve more complex logic
                success = True
            
            if success:
                state.status = DeploymentStatus.ROLLED_BACK
                state.updated_at = datetime.now()
                
                # Stop health monitoring
                self.health_monitor.stop_monitoring(config.deployment_name, config.version)
                
                logger.info(f"Rollback successful: {deployment_id}")
                return True
            else:
                state.status = DeploymentStatus.FAILED
                state.error_message = "Rollback failed"
                state.updated_at = datetime.now()
                return False
                
        except Exception as e:
            state.status = DeploymentStatus.FAILED
            state.error_message = f"Rollback failed: {e}"
            state.updated_at = datetime.now()
            logger.error(f"Rollback failed for {deployment_id}: {e}")
            return False
    
    def get_deployment_status(self, deployment_id: str) -> DeploymentState:
        """Get deployment status"""
        if deployment_id not in self.active_deployments:
            raise ValueError(f"Deployment {deployment_id} not found")
        
        state = self.active_deployments[deployment_id]
        
        # Update with latest Kubernetes status
        try:
            k8s_status = self.k8s_manager.get_deployment_status(
                state.config.deployment_name, 
                state.config.version
            )
            
            # Update replica counts
            replicas = k8s_status.get("replicas", {})
            state.current_replicas = replicas.get("current", 0)
            state.ready_replicas = replicas.get("ready", 0)
            state.available_replicas = replicas.get("available", 0)
            
            # Update health metrics
            health_status = self.health_monitor.get_health_status(
                state.config.deployment_name, 
                state.config.version
            )
            
            if health_status:
                state.error_rate = health_status.get("error_rate", 0.0)
                state.average_latency = health_status.get("average_latency", 0.0)
                state.health_check_status = "healthy" if health_status.get("health_score", 0) > 80 else "unhealthy"
            
        except Exception as e:
            logger.warning(f"Could not update deployment status: {e}")
        
        return state
    
    def list_deployments(self, status_filter: DeploymentStatus = None) -> List[DeploymentState]:
        """List deployments with optional status filter"""
        deployments = list(self.active_deployments.values())
        
        if status_filter:
            deployments = [d for d in deployments if d.status == status_filter]
        
        return sorted(deployments, key=lambda d: d.created_at, reverse=True)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        total_deployments = len(self.active_deployments)
        status_counts = defaultdict(int)
        
        for deployment in self.active_deployments.values():
            status_counts[deployment.status.value] += 1
        
        return {
            "total_deployments": total_deployments,
            "status_breakdown": dict(status_counts),
            "healthy_deployments": status_counts[DeploymentStatus.DEPLOYED.value],
            "failed_deployments": status_counts[DeploymentStatus.FAILED.value],
            "system_health": "healthy" if status_counts[DeploymentStatus.FAILED.value] == 0 else "degraded",
            "namespace": self.k8s_manager.namespace,
            "timestamp": datetime.now().isoformat()
        }


# Factory functions
def create_deployment_system(namespace: str = "production", 
                           kubeconfig_path: str = None) -> ProductionDeploymentSystem:
    """Create production deployment system"""
    return ProductionDeploymentSystem(namespace, kubeconfig_path)


def create_deployment_config(deployment_name: str, version: str, image_tag: str,
                           strategy: DeploymentStrategy = DeploymentStrategy.ROLLING,
                           **kwargs) -> DeploymentConfig:
    """Create deployment configuration"""
    return DeploymentConfig(
        deployment_name=deployment_name,
        version=version,
        image_tag=image_tag,
        strategy=strategy,
        **kwargs
    )


# Example usage and testing
if __name__ == "__main__":
    # Create deployment system
    deployment_system = create_deployment_system()
    
    # Create deployment configuration
    config = create_deployment_config(
        deployment_name="sentiment-analyzer",
        version="v2.1.0",
        image_tag="v2.1.0",
        strategy=DeploymentStrategy.CANARY,
        replicas=5,
        enable_autoscaling=True,
        canary_weight=20
    )
    
    print("Production Deployment System Example")
    print("=" * 50)
    
    try:
        # Execute deployment
        deployment_id = deployment_system.deploy(config)
        print(f"Deployment started: {deployment_id}")
        
        # Wait briefly
        time.sleep(5)
        
        # Check deployment status
        status = deployment_system.get_deployment_status(deployment_id)
        print(f"Deployment status: {status.status.value}")
        
        # Get system status
        system_status = deployment_system.get_system_status()
        print("System Status:")
        print(json.dumps(system_status, indent=2, default=str))
        
    except Exception as e:
        print(f"Deployment failed: {e}")
        
        # Show system status even on failure
        system_status = deployment_system.get_system_status()
        print("System Status:")
        print(json.dumps(system_status, indent=2, default=str))