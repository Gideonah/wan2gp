#!/usr/bin/env python3
"""
Download LTX-2 Distilled model files.

This downloads ONLY the files needed for LTX-2 Distilled image-to-video generation,
ignoring all Wan2.1 and other model families.

Total download size: ~20-25GB
- Transformer: ~10GB (fp8) or ~19GB (bf16)
- Text encoder (Gemma): ~2GB
- VAE + Vocoder + Upscaler: ~1GB

Usage:
    python download_ltx2_distilled.py [--fp8] [--output-dir /path/to/ckpts]
    
For Docker builds (vastai), use:
    python download_ltx2_distilled.py --fp8 --output-dir /workspace/ckpts
"""

import os
import sys
import argparse
from pathlib import Path

# Set up cache directory before imports
DEFAULT_CACHE_DIR = os.environ.get("HF_HOME", "/workspace/.cache/huggingface")
os.environ["HF_HOME"] = DEFAULT_CACHE_DIR

from huggingface_hub import snapshot_download, hf_hub_download


def parse_args():
    parser = argparse.ArgumentParser(description="Download LTX-2 Distilled models")
    parser.add_argument(
        "--fp8",
        action="store_true",
        help="Download FP8 quantized version (~10GB instead of ~19GB)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.environ.get("WAN2GP_CKPTS_DIR", None),
        help="Output directory (default: HF cache, or WAN2GP_CKPTS_DIR env var)",
    )
    parser.add_argument(
        "--no-gemma",
        action="store_true",
        help="Skip Gemma text encoder download",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    REPO_ID = "DeepBeepMeep/LTX-2"
    
    # Choose transformer based on precision
    if args.fp8:
        transformer_file = "ltx-2-19b-distilled-fp8_diffusion_model.safetensors"
        print("📦 Downloading LTX-2 Distilled FP8 (~10GB transformer)")
    else:
        transformer_file = "ltx-2-19b-distilled_diffusion_model.safetensors"
        print("📦 Downloading LTX-2 Distilled BF16 (~19GB transformer)")
    
    # Core model files (required)
    CORE_FILES = [
        # Main transformer
        transformer_file,
        
        # Video VAE encoder/decoder
        "ltx-2-19b_vae.safetensors",
        
        # Audio VAE (for audio generation)
        "ltx-2-19b_audio_vae.safetensors",
        
        # Vocoder (for audio output)
        "ltx-2-19b_vocoder.safetensors",
        
        # Text processing
        "ltx-2-19b_text_embedding_projection.safetensors",
        "ltx-2-19b-distilled_embeddings_connector.safetensors",
        
        # Spatial upscaler
        "ltx-2-spatial-upscaler-x2-1.0.safetensors",
    ]
    
    # Gemma text encoder files (in subfolder)
    GEMMA_FOLDER = "gemma-3-12b-it-qat-q4_0-unquantized"
    GEMMA_FILES = [
        f"{GEMMA_FOLDER}/added_tokens.json",
        f"{GEMMA_FOLDER}/chat_template.json",
        f"{GEMMA_FOLDER}/config_light.json",
        f"{GEMMA_FOLDER}/generation_config.json",
        f"{GEMMA_FOLDER}/preprocessor_config.json",
        f"{GEMMA_FOLDER}/processor_config.json",
        f"{GEMMA_FOLDER}/special_tokens_map.json",
        f"{GEMMA_FOLDER}/tokenizer.json",
        f"{GEMMA_FOLDER}/tokenizer.model",
        f"{GEMMA_FOLDER}/tokenizer_config.json",
        f"{GEMMA_FOLDER}/{GEMMA_FOLDER}.safetensors",
        f"{GEMMA_FOLDER}/{GEMMA_FOLDER}_quanto_bf16_int8.safetensors",
    ]
    
    # Build full pattern list
    all_patterns = CORE_FILES.copy()
    if not args.no_gemma:
        all_patterns.extend(GEMMA_FILES)
    
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
        print("✅ LTX-2 Distilled model downloaded successfully!")
        print()
        print("📋 Downloaded files:")
        for f in CORE_FILES:
            print(f"   ✓ {f}")
        if not args.no_gemma:
            print(f"   ✓ {GEMMA_FOLDER}/ (text encoder)")
        
    except KeyboardInterrupt:
        print("\n❌ Download cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

