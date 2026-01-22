#!/usr/bin/env bash
#
# Build Wan2GP Docker Image for Vast.ai Serverless
#
# This script builds and optionally pushes the Vast.ai-ready image.
# Models are downloaded at RUNTIME (not baked in).
#
# Features:
#   - Smaller image (~8GB without model weights)
#   - SKIP shared preprocessing models (SAM, DWPose, depth, etc.)
#   - No LoRAs downloaded
#   - Models downloaded on first startup (~15GB)
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
echo "  📦 Models downloaded at RUNTIME (smaller image)"
echo "  ⚡ SKIP shared preprocessing models (SAM, DWPose, depth, etc.)"
echo "  ❌ No LoRAs downloaded"
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

# Step 1: Build Docker image
echo "🐳 Step 1: Building Docker image..."
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
echo "    On-Start Script: /workspace/start_vastai.sh"
echo ""
echo "  ENVIRONMENT VARIABLES (set in Vast template):"
echo "    PYWORKER_REPO:  URL to your pyworker repo"
echo "    GCP_CLIENT_EMAIL, GCP_PRIVATE_KEY_B64, GCP_PROJECT_ID (for GCS uploads)"
echo ""
echo "  FIRST STARTUP:"
echo "    ⏳ Will download LTX-2 Distilled model (~15GB)"
echo "    💡 Mount persistent volume to /workspace/ckpts for faster restarts"
echo ""
echo "  WHAT'S EXCLUDED (faster startup):"
echo "    ❌ SAM (segmentation)"
echo "    ❌ DWPose (pose estimation)"
echo "    ❌ Depth Anything V2"
echo "    ❌ Wav2Vec, Roformer (audio preprocessing)"
echo "    ❌ LoRAs"
echo ""
echo "  ENDPOINTS:"
echo "    POST /generate/ltx2/i2v  - Image-to-Video generation"
echo "    GET  /health             - Health check"
echo "═══════════════════════════════════════════════════════════════════"
