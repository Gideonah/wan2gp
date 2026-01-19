#!/usr/bin/env bash
#
# Quick setup script for Wan2GP API on Vast.ai
# 
# Run this on a fresh Vast.ai instance:
#   curl -sSL https://raw.githubusercontent.com/YOUR_FORK/Wan2GP/main/setup_api_on_vast.sh | bash
#
# Or run locally after SSH-ing into the instance
#

set -e

echo "🚀 Setting up Wan2GP API Server..."

cd /workspace

# Clone if not already cloned
if [ ! -d "Wan2GP" ]; then
    echo "📦 Cloning Wan2GP..."
    git clone https://github.com/deepbeepmeep/Wan2GP.git
fi

cd Wan2GP

# Install Python dependencies
echo "📚 Installing Python dependencies..."
pip install -q -r requirements.txt
pip install -q fastapi uvicorn python-multipart

# Install PyTorch with CUDA (if not already installed)
echo "🔧 Installing PyTorch with CUDA..."
pip install -q torch==2.6.0+cu124 torchvision==0.21.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124

# Create output directory
mkdir -p /workspace/outputs

echo "✅ Setup complete!"
echo ""
echo "To start the API server, run:"
echo "  python api_server.py --model-type t2v --profile 5"
echo ""
echo "Or for image-to-video:"
echo "  python api_server.py --model-type i2v --profile 5"

