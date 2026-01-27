#!/usr/bin/env bash
#
# Wan2GP API Server Entrypoint
# 
# This entrypoint starts the FastAPI server instead of the Gradio UI,
# designed for serverless deployment on Vast.ai, RunPod, or Modal.
#
# Environment Variables:
#   WAN2GP_MODEL_TYPE: Model to load (default: t2v)
#   WAN2GP_PROFILE: MMGP profile 1-6 (default: 5)
#   WAN2GP_PORT: API port (default: 8000)
#   WAN2GP_OUTPUT_DIR: Output directory (default: /workspace/outputs)
#

export HOME=/home/user
export PYTHONUNBUFFERED=1
export HF_HOME=/home/user/.cache/huggingface

export OMP_NUM_THREADS=$(nproc)
export MKL_NUM_THREADS=$(nproc)
export OPENBLAS_NUM_THREADS=$(nproc)
export NUMEXPR_NUM_THREADS=$(nproc)

export TORCH_ALLOW_TF32_CUBLAS=1
export TORCH_ALLOW_TF32_CUDNN=1

# Disable audio warnings in Docker
export SDL_AUDIODRIVER=dummy
export PULSE_RUNTIME_PATH=/tmp/pulse-runtime

# API defaults
export WAN2GP_MODEL_TYPE=${WAN2GP_MODEL_TYPE:-"t2v"}
export WAN2GP_PROFILE=${WAN2GP_PROFILE:-"5"}
export WAN2GP_PORT=${WAN2GP_PORT:-"8000"}
export WAN2GP_OUTPUT_DIR=${WAN2GP_OUTPUT_DIR:-"/workspace/outputs"}

# ═══════════════════════════════════════════════════════════════════════════════
# CUDA DEBUG CHECKS
# ═══════════════════════════════════════════════════════════════════════════════

echo "🔍 CUDA Environment Debug Information:"
echo "═══════════════════════════════════════════════════════════════════════"

# Check CUDA driver on host (if accessible)
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "✅ nvidia-smi available"
    echo "📊 GPU Information:"
    nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv,noheader,nounits 2>/dev/null || echo "❌ nvidia-smi failed to query GPU"
else
    echo "❌ nvidia-smi not available in container"
fi

# Check PyTorch CUDA availability
echo ""
echo "🐍 PyTorch CUDA Check:"
python3 -c "
import sys
try:
    import torch
    print('✅ PyTorch imported successfully')
    print(f'   Version: {torch.__version__}')
    print(f'   CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'   CUDA version: {torch.version.cuda}')
        print(f'   Device count: {torch.cuda.device_count()}')
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f'   Device {i}: {props.name} ({props.total_memory//1024//1024}MB)')
    else:
        print('❌ CUDA not available to PyTorch')
        sys.exit(1)
except ImportError as e:
    print(f'❌ Failed to import PyTorch: {e}')
    sys.exit(1)
except Exception as e:
    print(f'❌ PyTorch CUDA check failed: {e}')
    sys.exit(1)
" 2>&1

if [ $? -ne 0 ]; then
    echo "❌ CUDA check failed. Exiting."
    exit 1
fi

# Create output directory
mkdir -p "$WAN2GP_OUTPUT_DIR"
chown -R user:user "$WAN2GP_OUTPUT_DIR" 2>/dev/null || true

echo "═══════════════════════════════════════════════════════════════════════"
echo "🚀 Starting Wan2GP API Server..."
echo "   Model Type: $WAN2GP_MODEL_TYPE"
echo "   Profile:    $WAN2GP_PROFILE"
echo "   Port:       $WAN2GP_PORT"
echo "   Output Dir: $WAN2GP_OUTPUT_DIR"
echo ""

# Start the API server
exec su -p user -c "python3 api_server.py --port $WAN2GP_PORT --model-type $WAN2GP_MODEL_TYPE --profile $WAN2GP_PROFILE $*"



