#!/bin/bash
# Build script for Sentiment Analyzer Pro
# Provides standardized build commands for different environments

set -euo pipefail

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="sentiment-analyzer-pro"
IMAGE_TAG="${1:-latest}"
BUILD_CONTEXT="."
DOCKERFILE="Dockerfile"

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

show_help() {
    cat << EOF
Sentiment Analyzer Pro Build Script

Usage: $0 [TAG] [OPTIONS]

Arguments:
  TAG                 Image tag (default: latest)

Options:
  --dev              Build development image
  --production       Build production image (default)
  --multi-arch       Build multi-architecture image
  --push             Push image to registry after build
  --no-cache         Build without using cache
  --help             Show this help message

Examples:
  $0                           # Build latest production image
  $0 v1.0.0                   # Build production image with v1.0.0 tag
  $0 --dev                    # Build development image
  $0 v1.0.0 --push           # Build and push v1.0.0 image
  $0 --multi-arch --push     # Build multi-arch and push

Environment Variables:
  REGISTRY           Docker registry URL (e.g., ghcr.io/owner)
  DOCKER_BUILDKIT    Enable BuildKit (default: 1)
EOF
}

check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi
}

build_image() {
    local tag="$1"
    local dockerfile="$2"
    local target="$3"
    local push="$4"
    local no_cache="$5"
    local multi_arch="$6"
    
    local full_image_name="${REGISTRY:+$REGISTRY/}${IMAGE_NAME}:${tag}"
    
    log_info "Building image: $full_image_name"
    log_info "Dockerfile: $dockerfile"
    log_info "Target: $target"
    
    # Build arguments
    local build_args=(
        "build"
        "--file" "$dockerfile"
    )
    
    if [[ -n "$target" ]]; then
        build_args+=("--target" "$target")
    fi
    
    if [[ "$no_cache" == "true" ]]; then
        build_args+=("--no-cache")
    fi
    
    if [[ "$multi_arch" == "true" ]]; then
        build_args+=(
            "--platform" "linux/amd64,linux/arm64"
            "--builder" "multiarch"
        )
        
        # Create builder if it doesn't exist
        if ! docker buildx inspect multiarch &> /dev/null; then
            log_info "Creating multi-arch builder"
            docker buildx create --name multiarch --use
        fi
        
        build_args[0]="buildx build"
        
        if [[ "$push" == "true" ]]; then
            build_args+=("--push")
        fi
    else
        build_args+=("--tag" "$full_image_name")
    fi
    
    build_args+=("$BUILD_CONTEXT")
    
    log_info "Running: docker ${build_args[*]}"
    
    # Execute build
    if docker "${build_args[@]}"; then
        log_info "Build completed successfully"
        
        # Push if requested and not multi-arch (multi-arch pushes automatically)
        if [[ "$push" == "true" && "$multi_arch" == "false" ]]; then
            log_info "Pushing image: $full_image_name"
            docker push "$full_image_name"
        fi
        
        # Show image info
        if [[ "$multi_arch" == "false" ]]; then
            log_info "Image size: $(docker images --format "table {{.Size}}" "$full_image_name" | tail -n1)"
        fi
        
    else
        log_error "Build failed"
        exit 1
    fi
}

run_security_scan() {
    local image_name="$1"
    
    log_info "Running security scan on $image_name"
    
    # Use trivy if available
    if command -v trivy &> /dev/null; then
        trivy image "$image_name"
    elif command -v docker &> /dev/null; then
        # Try docker scout if available
        if docker scout version &> /dev/null; then
            docker scout cves "$image_name"
        else
            log_warn "No security scanning tool available (trivy or docker scout)"
        fi
    fi
}

main() {
    # Default values
    local dev_mode="false"
    local production_mode="true"
    local multi_arch="false"
    local push="false"
    local no_cache="false"
    local run_scan="false"
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dev)
                dev_mode="true"
                production_mode="false"
                shift
                ;;
            --production)
                production_mode="true"
                dev_mode="false"
                shift
                ;;
            --multi-arch)
                multi_arch="true"
                shift
                ;;
            --push)
                push="true"
                shift
                ;;
            --no-cache)
                no_cache="true"
                shift
                ;;
            --scan)
                run_scan="true"
                shift
                ;;
            --help)
                show_help
                exit 0
                ;;
            -*)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
            *)
                IMAGE_TAG="$1"
                shift
                ;;
        esac
    done
    
    # Validate prerequisites
    check_docker
    
    # Enable BuildKit
    export DOCKER_BUILDKIT=1
    
    # Determine build configuration
    if [[ "$dev_mode" == "true" ]]; then
        DOCKERFILE="Dockerfile.dev"
        target=""
        log_info "Building development image"
    else
        DOCKERFILE="Dockerfile"
        target="production"
        log_info "Building production image"
    fi
    
    # Build image
    build_image "$IMAGE_TAG" "$DOCKERFILE" "$target" "$push" "$no_cache" "$multi_arch"
    
    # Run security scan if requested
    if [[ "$run_scan" == "true" ]]; then
        local full_image_name="${REGISTRY:+$REGISTRY/}${IMAGE_NAME}:${IMAGE_TAG}"
        run_security_scan "$full_image_name"
    fi
    
    log_info "Build process completed"
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi