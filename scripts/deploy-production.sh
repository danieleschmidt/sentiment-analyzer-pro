#!/bin/bash

# Production deployment script for Sentiment Analyzer
set -euo pipefail

# Configuration
DOCKER_REGISTRY="${DOCKER_REGISTRY:-sentiment-analyzer}"
VERSION="${VERSION:-latest}"
COMPOSE_FILE="docker-compose.production.yml"
ENV_FILE="${ENV_FILE:-.env.production}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    command -v docker >/dev/null 2>&1 || error "Docker is required but not installed"
    command -v docker-compose >/dev/null 2>&1 || error "Docker Compose is required but not installed"
    
    # Check if running as root or in docker group
    if [[ $EUID -eq 0 ]] || groups | grep -q docker; then
        log "Docker permissions verified"
    else
        error "Please run as root or add user to docker group"
    fi
    
    # Check if required files exist
    [[ -f "$COMPOSE_FILE" ]] || error "Compose file $COMPOSE_FILE not found"
    [[ -f "Dockerfile.production" ]] || error "Production Dockerfile not found"
}

# Create necessary directories
setup_directories() {
    log "Setting up directories..."
    
    mkdir -p logs/{nginx,app}
    mkdir -p models
    mkdir -p nginx/ssl
    mkdir -p monitoring/grafana/{dashboards,datasources}
    mkdir -p redis
    
    # Set proper permissions
    chmod 755 logs models nginx
    chmod 600 nginx/ssl/* 2>/dev/null || warn "SSL certificates not found"
}

# Generate SSL certificates if they don't exist
setup_ssl() {
    log "Setting up SSL certificates..."
    
    if [[ ! -f "nginx/ssl/cert.pem" ]] || [[ ! -f "nginx/ssl/key.pem" ]]; then
        warn "SSL certificates not found, generating self-signed certificates"
        openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
            -keyout nginx/ssl/key.pem \
            -out nginx/ssl/cert.pem \
            -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
        chmod 600 nginx/ssl/*
        log "Self-signed SSL certificates generated"
    else
        log "SSL certificates found"
    fi
}

# Build Docker images
build_images() {
    log "Building Docker images..."
    
    # Build the main application image
    docker build -f Dockerfile.production -t "${DOCKER_REGISTRY}:${VERSION}" .
    
    log "Docker images built successfully"
}

# Deploy with health checks
deploy() {
    log "Deploying application..."
    
    # Load environment variables if file exists
    if [[ -f "$ENV_FILE" ]]; then
        log "Loading environment from $ENV_FILE"
        export $(grep -v '^#' "$ENV_FILE" | xargs)
    fi
    
    # Pull latest images for third-party services
    docker-compose -f "$COMPOSE_FILE" pull nginx redis prometheus grafana
    
    # Deploy services
    docker-compose -f "$COMPOSE_FILE" up -d
    
    log "Services deployed, waiting for health checks..."
    
    # Wait for services to be healthy
    wait_for_health
    
    log "Deployment completed successfully!"
}

# Wait for services to become healthy
wait_for_health() {
    local max_attempts=30
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        log "Health check attempt $attempt/$max_attempts"
        
        # Check if sentiment analyzer is healthy
        if curl -f -s http://localhost/health >/dev/null 2>&1; then
            log "Application is healthy!"
            return 0
        fi
        
        if [[ $attempt -eq $max_attempts ]]; then
            error "Health check failed after $max_attempts attempts"
        fi
        
        sleep 10
        ((attempt++))
    done
}

# Run smoke tests
smoke_tests() {
    log "Running smoke tests..."
    
    # Test basic endpoints
    local base_url="https://localhost"
    
    # Test health endpoint
    if ! curl -k -f -s "$base_url/health" >/dev/null; then
        error "Health endpoint test failed"
    fi
    
    # Test prediction endpoint
    local prediction_test='{"text": "This is a test message"}'
    if ! curl -k -f -s -X POST -H "Content-Type: application/json" \
         -d "$prediction_test" "$base_url/predict" >/dev/null; then
        error "Prediction endpoint test failed"
    fi
    
    # Test metrics endpoint (might fail due to IP restrictions, that's ok)
    curl -k -f -s "$base_url/metrics" >/dev/null || warn "Metrics endpoint restricted (expected)"
    
    log "Smoke tests passed!"
}

# Show deployment status
show_status() {
    log "Deployment Status:"
    echo
    docker-compose -f "$COMPOSE_FILE" ps
    echo
    log "Application URLs:"
    echo "  - Application: https://localhost"
    echo "  - Grafana: http://localhost:3000 (admin/admin123)"
    echo "  - Prometheus: http://localhost:9090"
    echo
    log "Log locations:"
    echo "  - Application logs: ./logs/app/"
    echo "  - Nginx logs: ./logs/nginx/"
    echo
}

# Cleanup function
cleanup() {
    log "Cleaning up..."
    docker-compose -f "$COMPOSE_FILE" down
    docker system prune -f
}

# Rollback function
rollback() {
    log "Rolling back deployment..."
    docker-compose -f "$COMPOSE_FILE" down
    # Restore previous version if available
    if [[ -n "${PREVIOUS_VERSION:-}" ]]; then
        VERSION="$PREVIOUS_VERSION" deploy
    else
        error "No previous version available for rollback"
    fi
}

# Main deployment function
main() {
    local command="${1:-deploy}"
    
    case "$command" in
        "deploy")
            check_prerequisites
            setup_directories
            setup_ssl
            build_images
            deploy
            smoke_tests
            show_status
            ;;
        "cleanup")
            cleanup
            ;;
        "rollback")
            rollback
            ;;
        "status")
            show_status
            ;;
        "logs")
            docker-compose -f "$COMPOSE_FILE" logs -f "${2:-}"
            ;;
        *)
            echo "Usage: $0 {deploy|cleanup|rollback|status|logs [service]}"
            echo
            echo "Examples:"
            echo "  $0 deploy          # Full deployment"
            echo "  $0 status          # Show current status"
            echo "  $0 logs            # Show all logs"
            echo "  $0 logs nginx      # Show nginx logs"
            echo "  $0 cleanup         # Clean up deployment"
            echo "  $0 rollback        # Rollback to previous version"
            exit 1
            ;;
    esac
}

# Handle script interruption
trap cleanup EXIT

# Run main function
main "$@"