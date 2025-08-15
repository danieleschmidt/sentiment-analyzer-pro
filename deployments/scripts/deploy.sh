#!/bin/bash
set -e

echo "🚀 Deploying Sentiment Analyzer Pro to Production"

NAMESPACE=${NAMESPACE:-default}
CONFIG_DIR="deployments/configs"

# Apply Kubernetes manifests
echo "Applying Kubernetes manifests..."
kubectl apply -f $CONFIG_DIR/deployment.yaml -n $NAMESPACE
kubectl apply -f $CONFIG_DIR/service.yaml -n $NAMESPACE
kubectl apply -f $CONFIG_DIR/hpa.yaml -n $NAMESPACE

# Wait for rollout
echo "Waiting for deployment rollout..."
kubectl rollout status deployment/sentiment-analyzer-pro -n $NAMESPACE --timeout=300s

# Verify deployment
echo "Verifying deployment..."
kubectl get pods -l app=sentiment-analyzer-pro -n $NAMESPACE
kubectl get services -l app=sentiment-analyzer-pro -n $NAMESPACE

echo "✅ Deployment complete!"

# Optional: Run smoke tests
if [ "$RUN_SMOKE_TESTS" = "true" ]; then
    echo "Running smoke tests..."
    ./deployments/scripts/smoke_tests.sh
fi
