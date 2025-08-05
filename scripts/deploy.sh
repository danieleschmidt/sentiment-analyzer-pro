#!/bin/bash

# Production Deployment Script for Sentiment Analyzer Pro
# This script handles Docker and Kubernetes deployments with safety checks

set -euo pipefail

# Configuration
DOCKER_IMAGE="sentiment-analyzer-pro"
VERSION="${1:-latest}"
ENVIRONMENT="${ENVIRONMENT:-production}"
NAMESPACE="${NAMESPACE:-default}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check if running in production
    if [[ "$ENVIRONMENT" == "production" ]]; then
        log_warning "Deploying to PRODUCTION environment"
        read -p "Are you sure you want to continue? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Deployment cancelled"
            exit 0
        fi
    fi
    
    log_success "Prerequisites check passed"
}

# Build Docker image
build_image() {
    log_info "Building Docker image..."
    
    # Create models directory if it doesn't exist
    mkdir -p models
    
    # Train a basic model if none exists
    if [[ ! -f "models/sentiment_model.joblib" ]]; then
        log_info "Training basic model..."
        source venv/bin/activate 2>/dev/null || {
            log_warning "Virtual environment not found, using system Python"
        }
        python -m src.train --csv data/sample_reviews.csv --model models/sentiment_model.joblib
    fi
    
    # Build image
    docker build -t "${DOCKER_IMAGE}:${VERSION}" -t "${DOCKER_IMAGE}:latest" .
    
    log_success "Docker image built successfully"
}

# Run security scan
security_scan() {
    log_info "Running security scan..."
    
    # Scan the built image
    if command -v trivy &> /dev/null; then
        trivy image --severity HIGH,CRITICAL "${DOCKER_IMAGE}:${VERSION}"
    else
        log_warning "Trivy not found, skipping container security scan"
    fi
    
    log_success "Security scan completed"
}

# Deploy with Docker Compose
deploy_docker_compose() {
    log_info "Deploying with Docker Compose..."
    
    # Create necessary directories
    mkdir -p logs/nginx
    mkdir -p nginx/ssl
    
    # Generate self-signed certificates if they don't exist
    if [[ ! -f "nginx/ssl/cert.pem" ]]; then
        log_info "Generating self-signed SSL certificates..."
        openssl req -x509 -newkey rsa:4096 -keyout nginx/ssl/key.pem -out nginx/ssl/cert.pem \
            -days 365 -nodes -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
    fi
    
    # Deploy
    docker-compose -f docker-compose.production.yml up -d
    
    # Wait for services to be healthy
    log_info "Waiting for services to become healthy..."
    sleep 30
    
    # Health check
    if curl -f http://localhost/health &> /dev/null; then
        log_success "Deployment successful - services are healthy"
    else
        log_error "Deployment failed - health check failed"
        docker-compose -f docker-compose.production.yml logs
        exit 1
    fi
}

# Deploy to Kubernetes
deploy_kubernetes() {
    log_info "Deploying to Kubernetes..."
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed"
        exit 1
    fi
    
    # Apply manifests
    kubectl apply -f kubernetes/deployment.yaml -n "$NAMESPACE"
    
    # Wait for rollout
    kubectl rollout status deployment/sentiment-analyzer -n "$NAMESPACE" --timeout=300s
    
    # Get service status
    kubectl get pods,services,ingress -n "$NAMESPACE" -l app=sentiment-analyzer
    
    log_success "Kubernetes deployment successful"
}

# Rollback function
rollback() {
    log_warning "Rolling back deployment..."
    
    if [[ "$1" == "docker" ]]; then
        docker-compose -f docker-compose.production.yml down
        log_success "Docker Compose rollback completed"
    elif [[ "$1" == "kubernetes" ]]; then
        kubectl rollout undo deployment/sentiment-analyzer -n "$NAMESPACE"
        log_success "Kubernetes rollback completed"
    fi
}

# Monitoring setup
setup_monitoring() {
    log_info "Setting up monitoring..."
    
    # Check if monitoring is already running
    if docker-compose -f docker-compose.production.yml ps | grep -q prometheus; then
        log_info "Monitoring stack is running"
        log_info "Grafana: http://localhost:3000 (admin/admin)"
        log_info "Prometheus: http://localhost:9090"
    else
        log_warning "Monitoring stack is not running"
    fi
}

# Performance test
performance_test() {
    log_info "Running performance test..."
    
    # Simple load test
    if command -v hey &> /dev/null; then
        hey -n 1000 -c 10 -H "Content-Type: application/json" \
            -d '{"text": "This is a great product!"}' \
            http://localhost/predict
    else
        log_warning "hey tool not found, skipping performance test"
        log_info "You can install it with: go install github.com/rakyll/hey@latest"
    fi
}

# Main deployment function
main() {
    log_info "Starting deployment of Sentiment Analyzer Pro v${VERSION}"
    
    case "${2:-docker}" in
        "docker")
            check_prerequisites
            build_image
            security_scan
            deploy_docker_compose
            setup_monitoring
            ;;
        "kubernetes")
            check_prerequisites
            build_image
            security_scan
            deploy_kubernetes
            ;;
        "test")
            performance_test
            ;;
        "rollback")
            rollback "${3:-docker}"
            ;;
        *)
            echo "Usage: $0 <version> [docker|kubernetes|test|rollback] [deployment_type]"
            echo "  version: Docker image version tag"
            echo "  deployment: docker (default), kubernetes, test, rollback"
            echo "  deployment_type: docker or kubernetes (for rollback only)"
            echo ""
            echo "Examples:"
            echo "  $0 v1.0.0 docker        # Deploy with Docker Compose"
            echo "  $0 v1.0.0 kubernetes    # Deploy to Kubernetes"
            echo "  $0 latest test          # Run performance test"
            echo "  $0 latest rollback docker # Rollback Docker deployment"
            exit 1
            ;;
    esac
    
    log_success "Deployment completed successfully!"
}

# Trap for cleanup on script exit
trap 'log_info "Deployment script finished"' EXIT

# Run main function
main "$@"