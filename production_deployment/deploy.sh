#!/bin/bash
# Production Deployment Script for Sentiment Analyzer Pro

set -e

echo "🚀 Starting Production Deployment..."

# Build Docker image
echo "🔨 Building Docker image..."
docker build -t sentiment-analyzer:latest .

# Deploy with Docker Compose
echo "🌐 Deploying with Docker Compose..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 30

# Health check
echo "🔍 Running health checks..."
curl -f http://localhost/ || {
    echo "❌ Health check failed!"
    docker-compose logs
    exit 1
}

echo "✅ Deployment completed successfully!"
echo "🌐 Service available at: http://localhost/"
echo "📊 Health endpoint: http://localhost/"
echo "🔍 Logs: docker-compose logs -f"
