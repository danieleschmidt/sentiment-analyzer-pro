#!/bin/bash
set -e

echo "🚀 Deploying Sentiment Analyzer Pro"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running"
    exit 1
fi

# Stop existing containers
echo "Stopping existing containers..."
docker-compose -f docker-compose.production.yml down

# Pull latest images
echo "Pulling latest images..."
docker-compose -f docker-compose.production.yml pull

# Start services
echo "Starting services..."
docker-compose -f docker-compose.production.yml up -d

# Wait for services to be ready
echo "Waiting for services to be ready..."
sleep 30

# Health check
echo "Performing health check..."
curl -f http://localhost:5000/ || {
    echo "❌ Health check failed"
    docker-compose -f docker-compose.production.yml logs sentiment-analyzer
    exit 1
}

echo "✅ Deployment successful"
echo "🌐 Application is running at http://localhost:5000"
echo "📊 Prometheus is running at http://localhost:9090"
