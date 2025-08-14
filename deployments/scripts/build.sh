#!/bin/bash
set -e

echo "🚀 Building Sentiment Analyzer Pro for Production"

# Build production Docker image
echo "Building Docker image..."
docker build -f Dockerfile.production -t sentiment-analyzer-pro:latest .

# Tag with version
VERSION=$(date +%Y%m%d-%H%M%S)
docker tag sentiment-analyzer-pro:latest sentiment-analyzer-pro:$VERSION

echo "✅ Build complete: sentiment-analyzer-pro:$VERSION"

# Optional: Push to registry
if [ "$PUSH_TO_REGISTRY" = "true" ]; then
    echo "Pushing to registry..."
    docker push sentiment-analyzer-pro:latest
    docker push sentiment-analyzer-pro:$VERSION
fi
