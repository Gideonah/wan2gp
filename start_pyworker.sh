#!/usr/bin/env bash
#
# Vast.ai PyWorker Entrypoint for LTX-2 Distilled Video Generation
#
# This script:
#   1. Sets up the environment
#   2. Starts the LTX-2 API server (backend) with logs to file
#   3. Runs the PyWorker proxy
#
# Environment Variables:
#   WAN2GP_MODEL_TYPE: Model to load (default: ltx2_distilled)
#   WAN2GP_PROFILE: MMGP profile 1-6 (default: 5)
#   MODEL_SERVER_PORT: Backend API port (default: 8000)
#   WAN2GP_OUTPUT_DIR: Output directory (default: /workspace/outputs)
#   PYWORKER_PORT: PyWorker listening port (default: 80, set by Vast)
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

# API configuration (default to LTX-2 Distilled)
export WAN2GP_MODEL_TYPE=${WAN2GP_MODEL_TYPE:-"ltx2_distilled"}
export WAN2GP_PROFILE=${WAN2GP_PROFILE:-"3"}
export MODEL_SERVER_PORT=${MODEL_SERVER_PORT:-"8000"}
export WAN2GP_OUTPUT_DIR=${WAN2GP_OUTPUT_DIR:-"/workspace/outputs"}

# Log file location (PyWorker monitors this)
export WAN2GP_LOG_FILE="/var/log/wan2gp/server.log"
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
echo "🚀 Starting LTX-2 API Server (backend)..." | tee -a "$WAN2GP_LOG_FILE"
echo "   Model Type: $WAN2GP_MODEL_TYPE" | tee -a "$WAN2GP_LOG_FILE"
echo "   Profile:    $WAN2GP_PROFILE" | tee -a "$WAN2GP_LOG_FILE"
echo "   Port:       $MODEL_SERVER_PORT" | tee -a "$WAN2GP_LOG_FILE"
echo "   Output Dir: $WAN2GP_OUTPUT_DIR" | tee -a "$WAN2GP_LOG_FILE"
echo "" | tee -a "$WAN2GP_LOG_FILE"

# Change to app directory (code lives here, not /workspace which is a volume mount)
cd /app

# Start the API server in background, with all output going to log file
python3 api_server.py \
    --host 127.0.0.1 \
    --port "$MODEL_SERVER_PORT" \
    --model-type "$WAN2GP_MODEL_TYPE" \
    --profile "$WAN2GP_PROFILE" \
    >> "$WAN2GP_LOG_FILE" 2>&1 &

API_PID=$!
echo "📝 API server started (PID: $API_PID)" | tee -a "$WAN2GP_LOG_FILE"

# Give the server a moment to start (actual readiness is detected via logs)
sleep 2

# Verify process is still running
if ! kill -0 $API_PID 2>/dev/null; then
    echo "❌ API server failed to start! Check logs:" | tee -a "$WAN2GP_LOG_FILE"
    cat "$WAN2GP_LOG_FILE"
    exit 1
fi

# ═══════════════════════════════════════════════════════════════════════════════
# START PYWORKER
# ═══════════════════════════════════════════════════════════════════════════════

echo "" | tee -a "$WAN2GP_LOG_FILE"
echo "🌐 Starting Vast.ai PyWorker..." | tee -a "$WAN2GP_LOG_FILE"
echo "   Monitoring log: $WAN2GP_LOG_FILE" | tee -a "$WAN2GP_LOG_FILE"
echo "   Backend: http://127.0.0.1:$MODEL_SERVER_PORT" | tee -a "$WAN2GP_LOG_FILE"
echo "" | tee -a "$WAN2GP_LOG_FILE"

# Run the PyWorker (this blocks and handles incoming requests)
# PyWorker monitors the log file for readiness signals
exec python3 worker.py

