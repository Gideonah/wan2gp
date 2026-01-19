#!/usr/bin/env bash
#
# Bake Wan2GP Docker Image with Model Weights
#
# Run this script on a Vast.ai instance to:
# 1. Download model weights
# 2. Build a Docker image with weights included
# 3. Push to Docker Hub (or other registry)
#
# Prerequisites:
#   - Docker installed (Vast.ai instances have Docker)
#   - GPU available (for downloading weights)
#   - Docker Hub account (or other registry)
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

# Step 1: Download model weights first (faster than doing it in Docker build)
echo "📥 Step 1: Downloading model weights..."
mkdir -p /workspace/.cache/huggingface

python3 << 'EOF'
import os
os.environ['HF_HOME'] = '/workspace/.cache/huggingface'

from huggingface_hub import snapshot_download

print("Downloading Wan2.1 base model...")
snapshot_download(
    "DeepBeepMeep/Wan2.1",
    local_dir="/workspace/.cache/huggingface/hub/models--DeepBeepMeep--Wan2.1",
    local_dir_use_symlinks=False
)

print("✅ Model weights downloaded!")
EOF

# Step 2: Build Docker image
echo ""
echo "🐳 Step 2: Building Docker image..."

# Create a temporary Dockerfile that copies the pre-downloaded weights
cat > Dockerfile.baked.local << 'DOCKERFILE'
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/workspace/.cache/huggingface

RUN apt update && \
    apt install -y python3 python3-pip git wget curl libgl1 libglib2.0-0 ffmpeg && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy pre-downloaded weights FIRST (for layer caching)
COPY .cache/huggingface /workspace/.cache/huggingface/

# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt && \
    pip install fastapi uvicorn python-multipart && \
    pip install --extra-index-url https://download.pytorch.org/whl/cu124 \
        torch==2.6.0+cu124 torchvision==0.21.0+cu124

# Copy application code
COPY . /workspace/

# Create directories
RUN mkdir -p /workspace/outputs

ENV WAN2GP_MODEL_TYPE="t2v"
ENV WAN2GP_PROFILE="5"
ENV WAN2GP_PORT="8000"
ENV WAN2GP_OUTPUT_DIR="/workspace/outputs"

EXPOSE 8000

CMD ["python3", "api_server.py", "--host", "0.0.0.0", "--port", "8000"]
DOCKERFILE

# Copy the cache directory to the build context
cp -r /workspace/.cache .cache 2>/dev/null || true

docker build -f Dockerfile.baked.local -t ${FULL_IMAGE} .

# Cleanup
rm -f Dockerfile.baked.local
rm -rf .cache

echo "✅ Docker image built: ${FULL_IMAGE}"

# Step 3: Push to registry
echo ""
echo "📤 Step 3: Pushing to Docker Hub..."
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

