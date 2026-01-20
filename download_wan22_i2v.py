#!/usr/bin/env python3
"""
Download Wan2.2 Image-to-Video model files with Lightning v2 LoRAs.

This downloads the files needed for Wan2.2 I2V generation, optimized for
fast inference with Lightning v2 acceleration (4 steps).

Total download size: ~25-30GB
- Transformer HIGH: ~10GB (FP8 Lightning v2)
- Transformer LOW: ~10GB (FP8 Lightning v2)
- Text encoder (UMT5-XXL): ~4GB
- CLIP (XLM-RoBERTa): ~2GB
- VAE: ~500MB

Available model variants:
- Lightning v2 (default): Fast 4-step inference, enhanced prompt comprehension
- Standard: Higher quality, 20-50 steps

Usage:
    python download_wan22_i2v.py [--standard] [--output-dir /path/to/ckpts]
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
    parser = argparse.ArgumentParser(description="Download Wan2.2 I2V models")
    parser.add_argument(
        "--standard",
        action="store_true",
        help="Download standard model instead of Lightning v2 (slower but higher quality)",
    )
    parser.add_argument(
        "--quantized",
        action="store_true", 
        help="Download INT8 quantized version (smaller, requires less VRAM)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: HF cache)",
    )
    parser.add_argument(
        "--no-vae-upscaler",
        action="store_true",
        help="Skip VAE upscaler download (saves ~500MB)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Repository IDs
    WAN21_REPO = "DeepBeepMeep/Wan2.1"  # Contains shared files (CLIP, UMT5, VAE base)
    WAN22_REPO = "DeepBeepMeep/Wan2.2"  # Contains Wan2.2 specific models
    
    print("=" * 70)
    print("         WAN2.2 IMAGE-TO-VIDEO MODEL DOWNLOAD")
    print("=" * 70)
    
    # ═══════════════════════════════════════════════════════════════════════
    # TRANSFORMER FILES
    # ═══════════════════════════════════════════════════════════════════════
    
    if args.standard:
        print("\n📦 Mode: STANDARD (20-50 steps, highest quality)")
        if args.quantized:
            WAN22_TRANSFORMER_FILES = [
                # Standard model - INT8 quantized
                "wan2.2_image2video_14B_high_quanto_mbf16_int8.safetensors",
                "wan2.2_image2video_14B_low_quanto_mbf16_int8.safetensors",
            ]
            print("   Precision: INT8 quantized (~6GB each)")
        else:
            WAN22_TRANSFORMER_FILES = [
                # Standard model - BF16
                "wan2.2_image2video_14B_high_mbf16.safetensors",
                "wan2.2_image2video_14B_low_mbf16.safetensors",
            ]
            print("   Precision: BF16 full (~10GB each)")
    else:
        print("\n📦 Mode: LIGHTNING v2 (4 steps, fast + enhanced prompts)")
        # Lightning v2 Enhanced - Always FP8 optimized
        WAN22_TRANSFORMER_FILES = [
            "wan22EnhancedLightning_v2I2VFP8HIGH.safetensors",
            "wan22EnhancedLightning_v2I2VFP8LOW.safetensors",
        ]
        print("   Precision: FP8 optimized (~10GB each)")
    
    # ═══════════════════════════════════════════════════════════════════════
    # TEXT ENCODER FILES (from Wan2.1 repo - shared)
    # ═══════════════════════════════════════════════════════════════════════
    
    # UMT5-XXL text encoder
    UMT5_FOLDER = "umt5-xxl"
    if args.quantized:
        UMT5_MODEL = "models_t5_umt5-xxl-enc-quanto_int8.safetensors"
    else:
        UMT5_MODEL = "models_t5_umt5-xxl-enc-bf16.safetensors"
    
    UMT5_FILES = [
        f"{UMT5_FOLDER}/{UMT5_MODEL}",
        f"{UMT5_FOLDER}/special_tokens_map.json",
        f"{UMT5_FOLDER}/spiece.model",
        f"{UMT5_FOLDER}/tokenizer.json",
        f"{UMT5_FOLDER}/tokenizer_config.json",
    ]
    
    # XLM-RoBERTa CLIP encoder
    CLIP_FOLDER = "xlm-roberta-large"
    CLIP_FILES = [
        f"{CLIP_FOLDER}/models_clip_open-clip-xlm-roberta-large-vit-huge-14-bf16.safetensors",
        f"{CLIP_FOLDER}/sentencepiece.bpe.model",
        f"{CLIP_FOLDER}/special_tokens_map.json",
        f"{CLIP_FOLDER}/tokenizer.json",
        f"{CLIP_FOLDER}/tokenizer_config.json",
    ]
    
    # ═══════════════════════════════════════════════════════════════════════
    # VAE FILES
    # ═══════════════════════════════════════════════════════════════════════
    
    # Base VAE (from Wan2.1 - required)
    WAN21_VAE_FILES = [
        "Wan2.1_VAE.safetensors",
    ]
    
    # VAE upscaler (optional but recommended for 720p+)
    if not args.no_vae_upscaler:
        WAN21_VAE_FILES.append("Wan2.1_VAE_upscale2x_imageonly_real_v1.safetensors")
    
    # ═══════════════════════════════════════════════════════════════════════
    # DOWNLOAD EXECUTION
    # ═══════════════════════════════════════════════════════════════════════
    
    print(f"\n🔧 HF Cache: {DEFAULT_CACHE_DIR}")
    
    local_dir = args.output_dir if args.output_dir else None
    
    try:
        # Download from Wan2.1 repo (shared components)
        print(f"\n📂 Downloading from {WAN21_REPO}...")
        wan21_patterns = UMT5_FILES + CLIP_FILES + WAN21_VAE_FILES
        print(f"   Files: {len(wan21_patterns)}")
        
        snapshot_download(
            repo_id=WAN21_REPO,
            allow_patterns=wan21_patterns,
            local_dir=local_dir,
            local_dir_use_symlinks=False if local_dir else True,
            resume_download=True,
        )
        print("   ✓ Wan2.1 shared components downloaded")
        
        # Download from Wan2.2 repo (transformer models)
        print(f"\n📂 Downloading from {WAN22_REPO}...")
        print(f"   Files: {len(WAN22_TRANSFORMER_FILES)}")
        
        snapshot_download(
            repo_id=WAN22_REPO,
            allow_patterns=WAN22_TRANSFORMER_FILES,
            local_dir=local_dir,
            local_dir_use_symlinks=False if local_dir else True,
            resume_download=True,
        )
        print("   ✓ Wan2.2 transformer models downloaded")
        
        # Print summary
        print()
        print("=" * 70)
        print("✅ Wan2.2 I2V model downloaded successfully!")
        print("=" * 70)
        print()
        print("📋 Downloaded components:")
        print()
        print("   Transformers (Wan2.2):")
        for f in WAN22_TRANSFORMER_FILES:
            print(f"     ✓ {f}")
        print()
        print("   Text Encoders (shared):")
        print(f"     ✓ {UMT5_FOLDER}/ (UMT5-XXL)")
        print(f"     ✓ {CLIP_FOLDER}/ (XLM-RoBERTa CLIP)")
        print()
        print("   VAE (shared):")
        for f in WAN21_VAE_FILES:
            print(f"     ✓ {f}")
        
        print()
        print("💡 To use Wan2.2 I2V API, start the server with:")
        if args.standard:
            print("   python api_server.py --model-type i2v_2_2")
        else:
            print("   python api_server.py --model-type i2v_2_2_Enhanced_Lightning_v2")
        
        print()
        print("📝 Model Settings:")
        if args.standard:
            print("   - Inference steps: 20-50 (recommended: 30)")
            print("   - Guidance scale: 3.5")
            print("   - Use for: Maximum quality, complex scenes")
        else:
            print("   - Inference steps: 4 (fixed)")
            print("   - Guidance scale: 1.0")
            print("   - Features: Enhanced prompt comprehension, camera control")
            print("   - Use for: Fast iteration, production workloads")
        
    except KeyboardInterrupt:
        print("\n❌ Download cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

