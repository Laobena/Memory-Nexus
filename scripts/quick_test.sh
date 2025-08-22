#!/bin/bash
# Quick test script for development

set -e

echo "🔍 Quick Test Suite"
echo "=================="

# Build in dev-fast mode for quick iteration
echo "Building in dev-fast mode..."
cargo build --profile=dev-fast

# Run core tests only
echo "Running core tests..."
cargo test --profile=dev-fast --lib core:: --quiet

# Check if services are needed
if [ "$1" == "--with-services" ]; then
    echo "Starting services..."
    docker-compose up -d
    sleep 5
    
    echo "Testing health endpoint..."
    curl -sf http://localhost:8086/health || echo "Service not ready"
    
    docker-compose down
fi

echo "✅ Quick test complete!"