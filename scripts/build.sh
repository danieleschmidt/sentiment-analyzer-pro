#!/bin/bash
set -e

echo "🔨 Building Sentiment Analyzer Pro for Production"

# Build Docker image
echo "Building Docker image..."
docker build -f Dockerfile.production -t sentiment-analyzer-pro:latest .

# Tag with version
VERSION=$(date +%Y%m%d-%H%M%S)
docker tag sentiment-analyzer-pro:latest sentiment-analyzer-pro:$VERSION

echo "✅ Build complete: sentiment-analyzer-pro:latest, sentiment-analyzer-pro:$VERSION"
