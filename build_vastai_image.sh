#!/usr/bin/env bash
#
# Build Wan2GP Docker Image for Vast.ai Serverless
#
# This script builds and optionally pushes the Vast.ai-ready image
# with baked model weights and PyWorker integration.
#
# Usage:
#   ./build_vastai_image.sh YOUR_DOCKERHUB_USERNAME
#
# Example:
#   ./build_vastai_image.sh gideonah
#   # Creates: gideonah/wan2gp-vastai:latest
#

set -e

DOCKER_USERNAME=${1:-"your-username"}
IMAGE_NAME="wan2gp-vastai"
TAG="latest"
FULL_IMAGE="${DOCKER_USERNAME}/${IMAGE_NAME}:${TAG}"

echo "═══════════════════════════════════════════════════════════════════"
echo "  Wan2GP Vast.ai Serverless Image Builder"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "  Image: ${FULL_IMAGE}"
echo ""

# Check if we're in the right directory
if [ ! -f "api_server.py" ]; then
    echo "❌ Error: Run this script from the Wan2GP directory"
    exit 1
fi

if [ ! -f "Dockerfile.vastai" ]; then
    echo "❌ Error: Dockerfile.vastai not found"
    exit 1
fi

if [ ! -f "worker.py" ]; then
    echo "❌ Error: worker.py not found"
    exit 1
fi

# Step 1: Build Docker image
echo "🐳 Step 1: Building Docker image..."
echo "   (Model weights will be downloaded during build - this takes a while)"
echo ""

docker build -f Dockerfile.vastai -t ${FULL_IMAGE} .

echo ""
echo "✅ Docker image built: ${FULL_IMAGE}"

# Step 2: Push to registry
echo ""
echo "📤 Step 2: Pushing to Docker Hub..."
echo "   (Make sure you're logged in: docker login)"

read -p "Push to Docker Hub? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    docker push ${FULL_IMAGE}
    echo "✅ Image pushed: ${FULL_IMAGE}"
else
    echo "⏭️  Skipped push. To push later:"
    echo "   docker push ${FULL_IMAGE}"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  Done! Your Vast.ai-ready image: ${FULL_IMAGE}"
echo ""
echo "  VAST.AI TEMPLATE SETTINGS:"
echo "    Image:          ${FULL_IMAGE}"
echo "    Docker Options: --gpus all"
echo "    On-Start Script: /workspace/start_pyworker.sh"
echo ""
echo "  ENVIRONMENT VARIABLES (optional):"
echo "    WAN2GP_MODEL_TYPE: t2v, i2v, vace_14B (default: t2v)"
echo "    WAN2GP_PROFILE:    1-6 (default: 5)"
echo ""
echo "  ENDPOINTS:"
echo "    POST /generate/t2v  - Text-to-Video generation"
echo "    POST /generate/i2v  - Image-to-Video generation"
echo "    GET  /health        - Health check"
echo "═══════════════════════════════════════════════════════════════════"

