#!/usr/bin/env bash
#
# Vast.ai Serverless Entrypoint for WAN 2.2 Lightning Video Generation
#
# This script follows the STANDARD Vast.ai approach:
#   1. Sets up the environment
#   2. Starts the WAN API server (backend) in background
#   3. Runs the standard Vast.ai start_server.sh
#
# Required Environment Variables (set in Vast template):
#   PYWORKER_REPO: Git URL to your pyworker repo containing worker.py
#   CONTAINER_ID: Set automatically by Vast.ai
#
# Optional Environment Variables:
#   PYWORKER_REF: Git ref to checkout (default: main)
#   WAN2GP_MODEL_TYPE: Model to load (default: i2v_2_2_Enhanced_Lightning_v2)
#   WAN2GP_PROFILE: MMGP profile 1-6 (default: 3)
#   MODEL_SERVER_PORT: Backend API port (default: 8000)
#
# Sliding Window Settings (configurable via env vars):
#   WAN2GP_SLIDING_WINDOW_SIZE: Sliding window size (default: 81)
#   WAN2GP_SLIDING_WINDOW_OVERLAP: Overlap frames (default: 4)
#   WAN2GP_COLOR_CORRECTION_STRENGTH: Color correction (default: 1)
#   WAN2GP_TEMPORAL_UPSAMPLING: RIFE upsampling (default: rife2)
#
# GCP Credentials (for uploading videos to Cloud Storage):
#   GCP_PROJECT_ID, GCP_PRIVATE_KEY_ID, GCP_PRIVATE_KEY_B64, GCP_CLIENT_EMAIL, GCP_CLIENT_ID
#

set -e

# ═══════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT SETUP
# ═══════════════════════════════════════════════════════════════════════════════

export PYTHONUNBUFFERED=1
export HF_HOME=${HF_HOME:-"/workspace/.cache/huggingface"}

# Performance tuning
export OMP_NUM_THREADS=$(nproc)
export MKL_NUM_THREADS=$(nproc)
export OPENBLAS_NUM_THREADS=$(nproc)
export NUMEXPR_NUM_THREADS=$(nproc)

# CUDA optimizations
export TORCH_ALLOW_TF32_CUBLAS=1
export TORCH_ALLOW_TF32_CUDNN=1

# Disable audio warnings in Docker
export SDL_AUDIODRIVER=dummy
export PULSE_RUNTIME_PATH=/tmp/pulse-runtime

# API configuration (default to WAN 2.2 Enhanced Lightning v2 - NOT SVI2Pro)
export WAN2GP_MODEL_TYPE=${WAN2GP_MODEL_TYPE:-"i2v_2_2_Enhanced_Lightning_v2"}
export WAN2GP_PROFILE=${WAN2GP_PROFILE:-"3"}
export MODEL_SERVER_PORT=${MODEL_SERVER_PORT:-"8000"}
export WAN2GP_OUTPUT_DIR=${WAN2GP_OUTPUT_DIR:-"/workspace/outputs"}

# Sliding Window Settings (auto-enabled for duration > 5s)
export WAN2GP_SLIDING_WINDOW_SIZE=${WAN2GP_SLIDING_WINDOW_SIZE:-"81"}
export WAN2GP_SLIDING_WINDOW_OVERLAP=${WAN2GP_SLIDING_WINDOW_OVERLAP:-"4"}
export WAN2GP_COLOR_CORRECTION_STRENGTH=${WAN2GP_COLOR_CORRECTION_STRENGTH:-"1"}
export WAN2GP_TEMPORAL_UPSAMPLING=${WAN2GP_TEMPORAL_UPSAMPLING:-"rife2"}

# CRITICAL: Skip shared preprocessing model downloads (SAM, DWPose, depth, etc.)
# These are not needed for serverless I2V API - models are baked into the image
export WAN2GP_SKIP_SHARED_DOWNLOADS="1"

# Log file location (PyWorker monitors this)
export WAN2GP_LOG_FILE="/var/log/wan2gp/server.log"
export MODEL_LOG="$WAN2GP_LOG_FILE"  # For Vast's start_server.sh
LOG_DIR=$(dirname "$WAN2GP_LOG_FILE")

# ═══════════════════════════════════════════════════════════════════════════════
# DIRECTORY SETUP
# ═══════════════════════════════════════════════════════════════════════════════

mkdir -p "$WAN2GP_OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# Clear old logs (PyWorker should only see this run's logs)
: > "$WAN2GP_LOG_FILE"

# ═══════════════════════════════════════════════════════════════════════════════
# CUDA VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

echo "🔍 Verifying CUDA environment..."
echo "═══════════════════════════════════════════════════════════════════" | tee -a "$WAN2GP_LOG_FILE"

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "✅ nvidia-smi available" | tee -a "$WAN2GP_LOG_FILE"
    nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free \
        --format=csv,noheader,nounits 2>/dev/null | tee -a "$WAN2GP_LOG_FILE" || \
        echo "❌ nvidia-smi failed" | tee -a "$WAN2GP_LOG_FILE"
else
    echo "❌ nvidia-smi not available" | tee -a "$WAN2GP_LOG_FILE"
fi

# Quick PyTorch CUDA check
python3 -c "
import torch
print('✅ PyTorch CUDA:', 'available' if torch.cuda.is_available() else 'NOT AVAILABLE')
if torch.cuda.is_available():
    print(f'   GPU: {torch.cuda.get_device_name(0)}')
    print(f'   VRAM: {torch.cuda.get_device_properties(0).total_memory // 1024 // 1024}MB')
" 2>&1 | tee -a "$WAN2GP_LOG_FILE"

echo "═══════════════════════════════════════════════════════════════════" | tee -a "$WAN2GP_LOG_FILE"

# ═══════════════════════════════════════════════════════════════════════════════
# START BACKEND API SERVER
# ═══════════════════════════════════════════════════════════════════════════════

echo "" | tee -a "$WAN2GP_LOG_FILE"
echo "🚀 Starting WAN 2.2 SVI2Pro API Server (backend)..." | tee -a "$WAN2GP_LOG_FILE"
echo "   Model Type: $WAN2GP_MODEL_TYPE" | tee -a "$WAN2GP_LOG_FILE"
echo "   Profile:    $WAN2GP_PROFILE" | tee -a "$WAN2GP_LOG_FILE"
echo "   Port:       $MODEL_SERVER_PORT" | tee -a "$WAN2GP_LOG_FILE"
echo "   Output Dir: $WAN2GP_OUTPUT_DIR" | tee -a "$WAN2GP_LOG_FILE"
echo "" | tee -a "$WAN2GP_LOG_FILE"
echo "   SVI2Pro Sliding Window Settings:" | tee -a "$WAN2GP_LOG_FILE"
echo "   - Window Size:          $WAN2GP_SLIDING_WINDOW_SIZE frames" | tee -a "$WAN2GP_LOG_FILE"
echo "   - Overlap:              $WAN2GP_SLIDING_WINDOW_OVERLAP frames" | tee -a "$WAN2GP_LOG_FILE"
echo "   - Color Correction:     $WAN2GP_COLOR_CORRECTION_STRENGTH" | tee -a "$WAN2GP_LOG_FILE"
echo "   - Temporal Upsampling:  $WAN2GP_TEMPORAL_UPSAMPLING" | tee -a "$WAN2GP_LOG_FILE"
echo "" | tee -a "$WAN2GP_LOG_FILE"

# Change to workspace directory
cd /workspace

# Start the API server in background, with all output going to log file
python3 api_server.py \
    --host 0.0.0.0 \
    --port "$MODEL_SERVER_PORT" \
    --model-type "$WAN2GP_MODEL_TYPE" \
    --profile "$WAN2GP_PROFILE" \
    >> "$WAN2GP_LOG_FILE" 2>&1 &

API_PID=$!
echo "📝 API server started (PID: $API_PID)" | tee -a "$WAN2GP_LOG_FILE"

# Give the server a moment to start
sleep 2

# Verify process is still running
if ! kill -0 $API_PID 2>/dev/null; then
    echo "❌ API server failed to start! Check logs:" | tee -a "$WAN2GP_LOG_FILE"
    cat "$WAN2GP_LOG_FILE"
    exit 1
fi

# ═══════════════════════════════════════════════════════════════════════════════
# START PYWORKER VIA STANDARD VAST.AI SCRIPT
# ═══════════════════════════════════════════════════════════════════════════════

echo "" | tee -a "$WAN2GP_LOG_FILE"
echo "🌐 Starting Vast.ai PyWorker via standard start_server.sh..." | tee -a "$WAN2GP_LOG_FILE"
echo "   PYWORKER_REPO: ${PYWORKER_REPO:-'https://github.com/vast-ai/pyworker (default)'}" | tee -a "$WAN2GP_LOG_FILE"
echo "   PYWORKER_REF:  ${PYWORKER_REF:-'(default branch)'}" | tee -a "$WAN2GP_LOG_FILE"
echo "   CONTAINER_ID:  ${CONTAINER_ID:-'(not set - required!)'}" | tee -a "$WAN2GP_LOG_FILE"
echo "   MODEL_LOG:     $MODEL_LOG" | tee -a "$WAN2GP_LOG_FILE"
echo "" | tee -a "$WAN2GP_LOG_FILE"

# Download and run the standard Vast.ai PyWorker startup script
# This will:
#   1. Set up a venv with uv
#   2. Clone $PYWORKER_REPO (or default to vast-ai/pyworker)
#   3. Install requirements.txt from the repo
#   4. Install vastai-sdk
#   5. Set up SSL certificates
#   6. Run worker.py
# curl -sSL "https://raw.githubusercontent.com/vast-ai/pyworker/main/start_server.sh" -o /tmp/start_server.sh
# chmod +x /tmp/start_server.sh
# exec /tmp/start_server.sh

# For now, just keep the API server running
echo "🎬 WAN 2.2 SVI2Pro API Server is running on port $MODEL_SERVER_PORT" | tee -a "$WAN2GP_LOG_FILE"
echo "   API Endpoints:" | tee -a "$WAN2GP_LOG_FILE"
echo "   - POST /generate/wan22/i2v  - Image-to-Video with sliding window" | tee -a "$WAN2GP_LOG_FILE"
echo "   - GET  /health              - Health check" | tee -a "$WAN2GP_LOG_FILE"
echo "   - GET  /info                - API info" | tee -a "$WAN2GP_LOG_FILE"

# Wait for the API server process
wait $API_PID


