#!/usr/bin/env python3
"""
Download Z-Image model files.

This downloads the files needed for Z-Image text-to-image generation.

Total download size: ~8-10GB
- Transformer: ~6GB (bf16) or ~3GB (int8)
- Text encoder (Qwen3): ~2GB
- VAE: ~200MB

Usage:
    python download_zimage.py [--quantized] [--output-dir /path/to/ckpts]
"""

import os
import sys
import argparse
from pathlib import Path

# Set up cache directory before imports
DEFAULT_CACHE_DIR = os.environ.get("HF_HOME", "/workspace/.cache/huggingface")
os.environ["HF_HOME"] = DEFAULT_CACHE_DIR

from huggingface_hub import snapshot_download


def parse_args():
    parser = argparse.ArgumentParser(description="Download Z-Image models")
    parser.add_argument(
        "--quantized",
        action="store_true",
        help="Download INT8 quantized version (~3GB instead of ~6GB)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: HF cache)",
    )
    parser.add_argument(
        "--control",
        action="store_true",
        help="Also download Z-Image Control model for image-guided generation",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    REPO_ID = "DeepBeepMeep/Z-Image"
    
    # Core model files
    if args.quantized:
        transformer_file = "ZImageTurbo_quanto_bf16_int8.safetensors"
        print("📦 Downloading Z-Image Turbo INT8 (~3GB transformer)")
    else:
        transformer_file = "ZImageTurbo_bf16.safetensors"
        print("📦 Downloading Z-Image Turbo BF16 (~6GB transformer)")
    
    # Core files required for Z-Image
    CORE_FILES = [
        # Main transformer
        transformer_file,
        
        # VAE
        "ZImageTurbo_VAE_bf16.safetensors",
        "ZImageTurbo_VAE_bf16_config.json",
        
        # Scheduler config
        "ZImageTurbo_scheduler_config.json",
    ]
    
    # Qwen3 text encoder files (in subfolder)
    QWEN3_FOLDER = "Qwen3"
    QWEN3_FILES = [
        f"{QWEN3_FOLDER}/tokenizer.json",
        f"{QWEN3_FOLDER}/tokenizer_config.json",
        f"{QWEN3_FOLDER}/vocab.json",
        f"{QWEN3_FOLDER}/config.json",
        f"{QWEN3_FOLDER}/merges.txt",
    ]
    
    # Qwen3 model weights (choose based on quantization)
    if args.quantized:
        QWEN3_FILES.append(f"{QWEN3_FOLDER}/qwen3_quanto_bf16_int8.safetensors")
    else:
        QWEN3_FILES.append(f"{QWEN3_FOLDER}/qwen3_bf16.safetensors")
    
    # Optional: Control model for image-guided generation
    CONTROL_FILES = []
    if args.control:
        print("📦 Also downloading Z-Image Control model")
        CONTROL_FILES = [
            "ZImageTurboControl2_bf16.safetensors",
        ]
    
    # Build full pattern list
    all_patterns = CORE_FILES + QWEN3_FILES + CONTROL_FILES
    
    print(f"\n🔧 HF Cache: {DEFAULT_CACHE_DIR}")
    print(f"📂 Repository: {REPO_ID}")
    print(f"📄 Files to download: {len(all_patterns)}")
    print()
    
    # Download files
    try:
        print("⏳ Starting download...")
        
        local_dir = args.output_dir if args.output_dir else None
        
        snapshot_download(
            repo_id=REPO_ID,
            allow_patterns=all_patterns,
            local_dir=local_dir,
            local_dir_use_symlinks=False if local_dir else True,
            resume_download=True,
        )
        
        print()
        print("✅ Z-Image model downloaded successfully!")
        print()
        print("📋 Downloaded files:")
        for f in CORE_FILES:
            print(f"   ✓ {f}")
        print(f"   ✓ {QWEN3_FOLDER}/ (text encoder)")
        if args.control:
            for f in CONTROL_FILES:
                print(f"   ✓ {f}")
        
        print()
        print("💡 To use Z-Image API, start the server with:")
        print("   python api_server.py --model-type z_image")
        
    except KeyboardInterrupt:
        print("\n❌ Download cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()



