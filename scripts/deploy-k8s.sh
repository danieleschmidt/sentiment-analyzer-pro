#!/bin/bash
set -e

echo "☸️ Deploying to Kubernetes"

# Apply manifests
kubectl apply -f k8s/deployment.yaml

# Wait for rollout
kubectl rollout status deployment/sentiment-analyzer

# Get service info
kubectl get service sentiment-analyzer-service

echo "✅ Kubernetes deployment complete"
