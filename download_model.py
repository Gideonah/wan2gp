#!/usr/bin/env python3
"""
Download Wan2.1 model files for Docker build.
Only downloads files needed for t2v (text-to-video).
"""
import os
os.environ['HF_HOME'] = '/workspace/.cache/huggingface'

from huggingface_hub import snapshot_download

REPO_ID = "DeepBeepMeep/Wan2.1"
LOCAL_DIR = "/workspace/.cache/huggingface/hub/models--DeepBeepMeep--Wan2.1"

# Patterns for t2v model (quantized saves ~50% disk/memory)
T2V_PATTERNS = [
    # T2V Transformer (choose ONE)
    # "wan2.1_text2video_14B_mbf16.safetensors",              # ~14GB, best quality
    "wan2.1_text2video_14B_quanto_mbf16_int8.safetensors",    # ~7GB, quantized
    
    # CLIP Text Encoder
    "xlm-roberta-large/models_clip_open-clip-xlm-roberta-large-vit-huge-14-bf16.safetensors",
    "xlm-roberta-large/sentencepiece.bpe.model",
    "xlm-roberta-large/special_tokens_map.json",
    "xlm-roberta-large/tokenizer.json",
    "xlm-roberta-large/tokenizer_config.json",
    
    # T5 Text Encoder
    "umt5-xxl/special_tokens_map.json",
    "umt5-xxl/spiece.model",
    "umt5-xxl/tokenizer.json", 
    "umt5-xxl/tokenizer_config.json",
    "umt5-xxl-enc-quanto_int8.safetensors",
    
    # VAE
    "Wan2.1_VAE.safetensors",
]

if __name__ == "__main__":
    print("Downloading Wan2.1 t2v model (quantized, ~12GB total)...")
    
    snapshot_download(
        repo_id=REPO_ID,
        local_dir=LOCAL_DIR,
        allow_patterns=T2V_PATTERNS,
        local_dir_use_symlinks=False,
    )
    
    print("Wan2.1 t2v model downloaded!")


