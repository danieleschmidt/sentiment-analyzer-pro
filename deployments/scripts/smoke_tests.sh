#!/bin/bash
set -e

echo "🧪 Running Production Smoke Tests"

SERVICE_URL=${SERVICE_URL:-"http://localhost:5000"}

# Test health endpoint
echo "Testing health endpoint..."
if curl -f "$SERVICE_URL/health" > /dev/null 2>&1; then
    echo "✅ Health check passed"
else
    echo "❌ Health check failed"
    exit 1
fi

# Test prediction endpoint
echo "Testing prediction endpoint..."
RESPONSE=$(curl -s -X POST "$SERVICE_URL/predict" \
    -H "Content-Type: application/json" \
    -d '{"text": "This is a test review"}')

if echo "$RESPONSE" | grep -q "prediction"; then
    echo "✅ Prediction test passed"
else
    echo "❌ Prediction test failed"
    exit 1
fi

# Test metrics endpoint
echo "Testing metrics endpoint..."
if curl -f "$SERVICE_URL/metrics" > /dev/null 2>&1; then
    echo "✅ Metrics endpoint passed"
else
    echo "❌ Metrics endpoint failed"
    exit 1
fi

echo "✅ All smoke tests passed!"
