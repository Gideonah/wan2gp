#!/usr/bin/env bash
#
# Bake Wan2GP Docker Image with Model Weights
#
# This script builds the image using Dockerfile.baked.
# For faster builds, it pre-downloads weights before building.
#
# Usage:
#   ./bake_docker_image.sh YOUR_DOCKERHUB_USERNAME
#
# Example:
#   ./bake_docker_image.sh gideonah
#   # Creates: gideonah/wan2gp-api:baked
#

set -e

DOCKER_USERNAME=${1:-"your-username"}
IMAGE_NAME="wan2gp-api"
TAG="baked"
FULL_IMAGE="${DOCKER_USERNAME}/${IMAGE_NAME}:${TAG}"

echo "═══════════════════════════════════════════════════════════════════"
echo "  Wan2GP Docker Image Baking Script"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "  Image: ${FULL_IMAGE}"
echo ""

# Check if we're in the right directory
if [ ! -f "api_server.py" ]; then
    echo "❌ Error: Run this script from the Wan2GP directory"
    exit 1
fi

if [ ! -f "Dockerfile.baked" ]; then
    echo "❌ Error: Dockerfile.baked not found"
    exit 1
fi

# Step 1: Build Docker image using Dockerfile.baked
echo "🐳 Step 1: Building Docker image..."
echo "   (Model weights will be downloaded during build)"

docker build -f Dockerfile.baked -t ${FULL_IMAGE} .

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
echo "  Done! Your baked image: ${FULL_IMAGE}"
echo ""
echo "  To run on any GPU machine:"
echo "    docker run --gpus all -p 8000:8000 ${FULL_IMAGE}"
echo "═══════════════════════════════════════════════════════════════════"
