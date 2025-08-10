#!/bin/bash
# Autonomous SDLC Deployment Script
# Production deployment with zero-downtime and comprehensive monitoring

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEPLOYMENT_ENV="${DEPLOYMENT_ENV:-production}"
VERSION="${VERSION:-$(git rev-parse --short HEAD)}"
BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log "Checking deployment prerequisites..."
    
    local missing_tools=()
    
    # Check required tools
    for tool in docker docker-compose git curl; do
        if ! command -v "$tool" >/dev/null 2>&1; then
            missing_tools+=("$tool")
        fi
    done
    
    if [[ ${#missing_tools[@]} -ne 0 ]]; then
        error "Missing required tools: ${missing_tools[*]}"
        exit 1
    fi
    
    # Check Docker daemon
    if ! docker info >/dev/null 2>&1; then
        error "Docker daemon is not running"
        exit 1
    fi
    
    # Check disk space (need at least 5GB)
    local available_space
    available_space=$(df / | awk 'NR==2 {print $4}')
    if [[ $available_space -lt 5242880 ]]; then  # 5GB in KB
        warn "Low disk space. Available: $((available_space / 1024 / 1024))GB"
    fi
    
    success "Prerequisites check passed"
}

# Pre-deployment quality gates
run_quality_gates() {
    log "Running pre-deployment quality gates..."
    
    cd "$PROJECT_ROOT"
    
    # Create virtual environment if it doesn't exist
    if [[ ! -d "venv" ]]; then
        python3 -m venv venv
    fi
    
    source venv/bin/activate
    
    # Install dependencies
    pip install -e . >/dev/null 2>&1
    pip install pytest >/dev/null 2>&1
    
    # Run tests
    log "Running comprehensive test suite..."
    if python -m pytest tests/test_autonomous_sdlc.py -v --tb=short; then
        success "All tests passed"
    else
        error "Tests failed - deployment aborted"
        exit 1
    fi
    
    # Run security scan if bandit is available
    if command -v bandit >/dev/null 2>&1; then
        log "Running security scan..."
        bandit -r src/ -f json -o bandit_report.json || warn "Security scan found issues"
    fi
    
    # Check code quality
    log "Checking code quality..."
    if command -v ruff >/dev/null 2>&1; then
        ruff check src/ || warn "Code quality issues found"
    fi
    
    success "Quality gates passed"
}

# Build application images
build_images() {
    log "Building production Docker images..."
    
    cd "$PROJECT_ROOT"
    
    # Build main application image
    docker build \
        -f docker/Dockerfile.production \
        -t "sentiment-analyzer-pro:${VERSION}" \
        -t "sentiment-analyzer-pro:latest" \
        --build-arg BUILD_DATE="$BUILD_DATE" \
        --build-arg VERSION="$VERSION" \
        --build-arg VCS_REF="$(git rev-parse HEAD)" \
        . || {
        error "Failed to build application image"
        exit 1
    }
    
    success "Docker images built successfully"
}

# Deploy application
deploy_application() {
    log "Deploying application to $DEPLOYMENT_ENV environment..."
    
    cd "$PROJECT_ROOT"
    
    # Set environment variables
    export BUILD_DATE VERSION
    export JWT_SECRET="${JWT_SECRET:-$(openssl rand -base64 32)}"
    export REDIS_PASSWORD="${REDIS_PASSWORD:-$(openssl rand -base64 16)}"
    export GRAFANA_PASSWORD="${GRAFANA_PASSWORD:-admin123}"
    
    # Create required directories
    mkdir -p logs models cache
    
    # Deploy with Docker Compose
    if [[ "$DEPLOYMENT_ENV" == "production" ]]; then
        log "Starting production deployment..."
        docker-compose -f docker-compose.production.yml up -d --remove-orphans
    else
        log "Starting development deployment..."
        docker-compose up -d --remove-orphans
    fi
    
    success "Application deployed successfully"
}

# Health check and validation
validate_deployment() {
    log "Validating deployment health..."
    
    local max_attempts=30
    local attempt=1
    local health_endpoint="http://localhost/health"
    
    if [[ "$DEPLOYMENT_ENV" != "production" ]]; then
        health_endpoint="http://localhost:5000/health"
    fi
    
    log "Waiting for services to start..."
    sleep 10
    
    while [[ $attempt -le $max_attempts ]]; do
        if curl -f -s "$health_endpoint" >/dev/null 2>&1; then
            success "Health check passed on attempt $attempt"
            break
        fi
        
        if [[ $attempt -eq $max_attempts ]]; then
            error "Health check failed after $max_attempts attempts"
            log "Checking service logs..."
            docker-compose logs --tail=50
            exit 1
        fi
        
        log "Health check attempt $attempt/$max_attempts failed, retrying..."
        sleep 5
        ((attempt++))
    done
    
    # Test API endpoints
    log "Testing API endpoints..."
    
    local api_endpoint="http://localhost/predict"
    if [[ "$DEPLOYMENT_ENV" != "production" ]]; then
        api_endpoint="http://localhost:5000/predict"
    fi
    
    # Test prediction endpoint
    local test_response
    test_response=$(curl -s -X POST "$api_endpoint" \
        -H "Content-Type: application/json" \
        -d '{"text": "This is a great test!"}' || echo "")
    
    if echo "$test_response" | grep -q "sentiment"; then
        success "API endpoints are working correctly"
    else
        error "API endpoint test failed"
        log "Response: $test_response"
        exit 1
    fi
    
    success "Deployment validation completed"
}

# Performance baseline
run_performance_test() {
    log "Running performance baseline test..."
    
    local api_endpoint="http://localhost/predict"
    if [[ "$DEPLOYMENT_ENV" != "production" ]]; then
        api_endpoint="http://localhost:5000/predict"
    fi
    
    # Simple load test
    log "Testing single prediction latency..."
    local start_time end_time duration
    start_time=$(date +%s%3N)
    
    curl -s -X POST "$api_endpoint" \
        -H "Content-Type: application/json" \
        -d '{"text": "Performance test text"}' >/dev/null
    
    end_time=$(date +%s%3N)
    duration=$((end_time - start_time))
    
    if [[ $duration -lt 1000 ]]; then
        success "Latency test passed: ${duration}ms"
    else
        warn "Latency test: ${duration}ms (target: <1000ms)"
    fi
    
    # Test batch processing
    log "Testing batch processing..."
    local batch_payload='{"texts": ["Great!", "Terrible!", "Okay", "Amazing!", "Poor"]}'
    
    start_time=$(date +%s%3N)
    curl -s -X POST "${api_endpoint}/batch" \
        -H "Content-Type: application/json" \
        -d "$batch_payload" >/dev/null
    end_time=$(date +%s%3N)
    duration=$((end_time - start_time))
    
    if [[ $duration -lt 2000 ]]; then
        success "Batch processing test passed: ${duration}ms"
    else
        warn "Batch processing test: ${duration}ms (target: <2000ms)"
    fi
}

# Deployment summary
deployment_summary() {
    log "Deployment Summary"
    echo "===================="
    echo "Environment: $DEPLOYMENT_ENV"
    echo "Version: $VERSION"
    echo "Build Date: $BUILD_DATE"
    echo "Git Commit: $(git rev-parse HEAD)"
    echo ""
    
    if [[ "$DEPLOYMENT_ENV" == "production" ]]; then
        echo "Services:"
        echo "  - Application: http://localhost"
        echo "  - Monitoring: http://localhost:3000 (Grafana)"
        echo "  - Metrics: http://localhost:9090 (Prometheus)"
    else
        echo "Services:"
        echo "  - Application: http://localhost:5000"
        echo "  - Health: http://localhost:5000/health"
        echo "  - Metrics: http://localhost:5000/metrics"
    fi
    
    echo ""
    log "Deployment completed successfully! 🎉"
}

# Rollback function
rollback_deployment() {
    log "Rolling back deployment..."
    
    cd "$PROJECT_ROOT"
    
    if [[ -f "docker-compose.backup.yml" ]]; then
        mv docker-compose.backup.yml docker-compose.yml
        docker-compose up -d --remove-orphans
        success "Rollback completed"
    else
        docker-compose down
        warn "No backup configuration found, services stopped"
    fi
}

# Signal handlers
trap 'error "Deployment interrupted"; exit 1' INT TERM

# Main deployment flow
main() {
    log "Starting Autonomous SDLC Deployment"
    log "Environment: $DEPLOYMENT_ENV"
    log "Version: $VERSION"
    
    # Backup current configuration if it exists
    if [[ -f "$PROJECT_ROOT/docker-compose.yml" ]] && [[ "$DEPLOYMENT_ENV" == "production" ]]; then
        cp "$PROJECT_ROOT/docker-compose.yml" "$PROJECT_ROOT/docker-compose.backup.yml"
    fi
    
    check_prerequisites
    run_quality_gates
    build_images
    deploy_application
    validate_deployment
    run_performance_test
    deployment_summary
}

# Handle command line arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "rollback")
        rollback_deployment
        ;;
    "validate")
        validate_deployment
        ;;
    "test")
        run_performance_test
        ;;
    *)
        echo "Usage: $0 {deploy|rollback|validate|test}"
        exit 1
        ;;
esac