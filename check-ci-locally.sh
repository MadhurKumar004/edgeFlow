#!/bin/bash
# Local CI checks - Run this before pushing to GitHub
# This mimics what GitHub Actions will run

set -e  # Exit on first error

echo "🔍 Running local CI checks..."
echo ""

# 1. Black formatting check
echo "📝 Checking code formatting with Black..."
black --check .
echo "✅ Black formatting passed!"
echo ""

# 2. Type checking with mypy
echo "🔎 Running type check with mypy..."
mypy --ignore-missing-imports .
echo "✅ Mypy type check passed!"
echo ""

# 3. Run tests with pytest
echo "🧪 Running pytest..."
PYTHONPATH=. python -m pytest -q tests --cov=src/edgeflow --cov-report=xml --cov-fail-under=75
echo "✅ Pytest passed!"
echo ""

# 4. Docker build test (optional - takes longer)
read -p "🐳 Test Docker builds? This takes ~5 minutes (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Building backend Docker image..."
    docker build -t edgeflow-backend:test -f src/edgeflow/backend/Dockerfile .
    echo "Building frontend Docker image..."
    docker build -t edgeflow-frontend:test -f src/edgeflow/frontend/Dockerfile src/edgeflow/frontend/
    echo "✅ Docker builds passed!"
fi

echo ""
echo "🎉 All CI checks passed! Safe to push to GitHub."
