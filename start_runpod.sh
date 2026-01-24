#!/usr/bin/env bash
#
# RunPod Startup Script for Wan2GP
#
# This script can be used as the entrypoint for RunPod deployments.
# It supports both Pod mode (FastAPI server) and Serverless mode (handler).
#
# Usage:
#   ./start_runpod.sh           # Default: Pod mode (FastAPI on port 8000)
#   ./start_runpod.sh serverless # Serverless handler mode
#

set -e

export PYTHONUNBUFFERED=1
export HF_HOME=/workspace/.cache/huggingface

# Model defaults
export WAN2GP_MODEL_TYPE=${WAN2GP_MODEL_TYPE:-"ltx2_distilled"}
export WAN2GP_PROFILE=${WAN2GP_PROFILE:-"3"}
export WAN2GP_OUTPUT_DIR=${WAN2GP_OUTPUT_DIR:-"/workspace/outputs"}
export PORT=${PORT:-8000}

# Create directories
mkdir -p "$WAN2GP_OUTPUT_DIR" /workspace/.cache/huggingface

echo "═══════════════════════════════════════════════════════════════════════"
echo "  Wan2GP for RunPod"
echo "═══════════════════════════════════════════════════════════════════════"
echo "  Model Type: $WAN2GP_MODEL_TYPE"
echo "  Profile:    $WAN2GP_PROFILE"
echo "  Output Dir: $WAN2GP_OUTPUT_DIR"
echo ""

# Check GPU
echo "🔍 GPU Check:"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || echo "❌ nvidia-smi query failed"
else
    echo "❌ nvidia-smi not available"
fi

echo ""

# Check mode
MODE=${1:-pod}

cd /workspace/wan2gp

if [ "$MODE" = "serverless" ]; then
    echo "🚀 Starting RunPod Serverless Handler..."
    exec python -u /handler.py
else
    echo "🚀 Starting FastAPI Server on port $PORT..."
    exec python api_server.py --model-type "$WAN2GP_MODEL_TYPE" --port "$PORT"
fi

