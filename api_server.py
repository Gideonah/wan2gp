#!/usr/bin/env python3
"""
Wan2GP Multi-Model API Server for Serverless Deployment

This FastAPI server provides a REST API interface to multiple generation models:
- Z-Image: Text-to-Image generation
- LTX-2 Distilled: Image-to-Video generation (fast, with audio)
- Wan2.2 I2V Lightning v2: Image-to-Video generation (4 steps, enhanced prompts)

Designed for serverless GPU deployments (Vast.ai, RunPod, Modal).

Usage:
    python api_server.py [--port 8000] [--model-type ltx2_distilled|z_image|i2v_2_2_lightning_v2]

Environment Variables:
    WAN2GP_MODEL_TYPE: Default model type to load
    WAN2GP_PROFILE: MMGP profile for memory optimization (1-6, default: 5)
    WAN2GP_OUTPUT_DIR: Directory to save generated outputs
"""

import os
import sys
import time
import uuid
import gc
import asyncio
import argparse
import httpx
from pathlib import Path
from typing import Optional, List, Union, Literal
from contextlib import asynccontextmanager
import base64
import io
import traceback

# ═══════════════════════════════════════════════════════════════════════════════
# CRITICAL: Parse our arguments BEFORE importing wgp.py
# wgp.py parses sys.argv at import time, so we must handle this first
# ═══════════════════════════════════════════════════════════════════════════════

def parse_api_args():
    """Parse API server arguments before wgp.py can interfere"""
    parser = argparse.ArgumentParser(description="Wan2GP API Server", add_help=False)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-type", type=str, default=os.environ.get("WAN2GP_MODEL_TYPE", "ltx2_19B"))
    parser.add_argument("--profile", type=int, default=int(os.environ.get("WAN2GP_PROFILE", "5")))
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("-h", "--help", action="store_true")
    
    args, unknown = parser.parse_known_args()
    
    if args.help:
        print("""
Wan2GP Multi-Model API Server

Usage:
    python api_server.py [OPTIONS]

Options:
    --host          Host to bind to (default: 0.0.0.0)
    --port          Port to listen on (default: 8000)
    --model-type    Model type to load:
                      - ltx2_distilled (default) - LTX-2 Image-to-Video
                      - z_image - Z-Image Text-to-Image  
                      - i2v_2_2_Enhanced_Lightning_v2 - Wan2.2 I2V Lightning v2
    --profile       MMGP memory profile 1-6 (default: 5)
    --reload        Enable auto-reload for development

Environment Variables:
    WAN2GP_MODEL_TYPE   Default model type
    WAN2GP_PROFILE      Default profile
    WAN2GP_OUTPUT_DIR   Output directory
        """)
        sys.exit(0)
    
    return args

# Parse our args first
_api_args = parse_api_args()

# Store our settings before clearing argv
API_HOST = _api_args.host
API_PORT = _api_args.port
API_MODEL_TYPE = _api_args.model_type
API_PROFILE = _api_args.profile
API_RELOAD = _api_args.reload

# Clear sys.argv so wgp.py's parser doesn't see our arguments
sys.argv = [sys.argv[0]]

# Add the Wan2GP root to path
ROOT_DIR = Path(__file__).parent.absolute()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Set environment variables before imports
os.environ["GRADIO_LANG"] = "en"

# Optimize CUDA memory allocation to reduce fragmentation
if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn

# Now it's safe to import from wgp/mmgp
from mmgp import offload
from shared.utils.audio_video import save_video
from shared.utils.loras_mutipliers import parse_loras_multipliers
from shared.attention import get_attention_modes, get_supported_attention_modes

# ═══════════════════════════════════════════════════════════════════════════════
# ATTENTION MODE DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def detect_best_attention_mode() -> str:
    """
    Detect the best available attention mode, preferring sage2.
    Returns the attention mode to use and logs the result.
    """
    installed = get_attention_modes()
    supported = get_supported_attention_modes()
    
    print("🔍 Detecting attention mode...")
    print(f"   Installed attention modes: {installed}")
    print(f"   Supported attention modes: {supported}")
    
    # Try sage2 first (best performance)
    if "sage2" in supported:
        print("✅ Using SageAttention 2 (sage2) - best performance")
        return "sage2"
    elif "sage2" in installed:
        print("⚠️  SageAttention 2 is installed but NOT supported on this GPU")
    
    # Try sage (original SageAttention)
    if "sage" in supported:
        print("✅ Using SageAttention 1 (sage)")
        return "sage"
    
    # Try flash attention
    if "flash" in supported:
        print("✅ Using Flash Attention")
        return "flash"
    
    # Fallback to SDPA (always available)
    print("⚠️  Falling back to SDPA attention (default PyTorch)")
    return "sdpa"

# Detect attention mode at import time
DETECTED_ATTENTION_MODE = detect_best_attention_mode()

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_MODEL_TYPE = API_MODEL_TYPE
DEFAULT_PROFILE = API_PROFILE
OUTPUT_DIR = Path(os.environ.get("WAN2GP_OUTPUT_DIR", "/workspace/outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# GCS CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "serverless_media_outputs")
GCS_ENABLED = os.environ.get("GCS_ENABLED", "true").lower() == "true"
GCS_PROJECT_ID = os.environ.get("GCP_PROJECT_ID", None)
# URL expiration in days for signed URLs
GCS_URL_EXPIRATION_DAYS = int(os.environ.get("GCS_URL_EXPIRATION_DAYS", "7"))

# GCS client (lazy initialized)
_gcs_client = None

def _get_gcp_credentials():
    """
    Build GCP credentials from environment variables (in-memory, no file written to disk).
    
    Required env vars:
        GCP_PROJECT_ID: Your GCP project ID
        GCP_CLIENT_EMAIL: Service account email (xxx@project.iam.gserviceaccount.com)
        GCP_PRIVATE_KEY_B64: Base64-encoded private key
    
    Optional env vars:
        GCP_PRIVATE_KEY_ID: Private key ID (can be empty)
        GCP_CLIENT_ID: Client ID (can be empty)
    
    Returns:
        google.oauth2.service_account.Credentials or None
    """
    import base64
    
    client_email = os.environ.get("GCP_CLIENT_EMAIL")
    private_key_b64 = os.environ.get("GCP_PRIVATE_KEY_B64")
    
    if not client_email or not private_key_b64:
        return None
    
    try:
        from google.oauth2 import service_account
        
        # Decode the base64-encoded private key
        private_key = base64.b64decode(private_key_b64).decode('utf-8')
        
        # Convert literal \n to actual newlines (JSON escaping issue)
        private_key = private_key.replace('\\n', '\n')
        
        # Build the credentials dict (in memory - never written to disk)
        credentials_info = {
            "type": "service_account",
            "project_id": os.environ.get("GCP_PROJECT_ID", ""),
            "private_key_id": os.environ.get("GCP_PRIVATE_KEY_ID", ""),
            "private_key": private_key,
            "client_email": client_email,
            "client_id": os.environ.get("GCP_CLIENT_ID", "103702167834083521665"),
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_x509_cert_url": f"https://www.googleapis.com/robot/v1/metadata/x509/{client_email}",
            "universe_domain": "googleapis.com"
        }
        
        # Create credentials object directly from dict (no file needed)
        credentials = service_account.Credentials.from_service_account_info(credentials_info)
        print("✅ GCP credentials loaded from env vars (in-memory, no file written)")
        return credentials
        
    except Exception as e:
        print(f"⚠️ Failed to build GCP credentials: {e}")
        return None

def get_gcs_client():
    """Get or create GCS client"""
    global _gcs_client
    if _gcs_client is None:
        try:
            from google.cloud import storage
            
            # Try env var credentials first (in-memory, no file written)
            credentials = _get_gcp_credentials()
            
            if credentials:
                _gcs_client = storage.Client(project=GCS_PROJECT_ID, credentials=credentials)
                print(f"✅ GCS client initialized for bucket: {GCS_BUCKET_NAME}")
            elif os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
                # Fall back to file-based credentials
                _gcs_client = storage.Client(project=GCS_PROJECT_ID)
                print(f"✅ GCS client initialized from GOOGLE_APPLICATION_CREDENTIALS")
            else:
                # Try Application Default Credentials
                _gcs_client = storage.Client(project=GCS_PROJECT_ID)
                print(f"✅ GCS client initialized with ADC")
                
        except Exception as e:
            print(f"⚠️ Failed to initialize GCS client: {e}")
            return None
    return _gcs_client

def upload_to_gcs(local_path: str, gcs_filename: str = None, content_type: str = "video/mp4") -> tuple[bool, str, str]:
    """
    Upload a file to GCS and return a signed URL.
    
    Args:
        local_path: Path to the local file
        gcs_filename: Optional filename in GCS (defaults to local filename)
        content_type: MIME type of the file
        
    Returns:
        tuple: (success, gcs_uri or signed_url, error_message)
    """
    if not GCS_ENABLED:
        return False, None, "GCS upload disabled"
    
    client = get_gcs_client()
    if client is None:
        return False, None, "GCS client not available"
    
    try:
        from datetime import timedelta
        
        local_path = Path(local_path)
        if not local_path.exists():
            return False, None, f"File not found: {local_path}"
        
        # Use provided filename or extract from path
        filename = gcs_filename or local_path.name
        gcs_path = f"videos/{filename}"
        
        bucket = client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(gcs_path)
        
        # Set chunk size for large files
        blob.chunk_size = 8 * 1024 * 1024  # 8MB chunks
        
        # Upload with retry
        print(f"📤 Uploading to GCS: gs://{GCS_BUCKET_NAME}/{gcs_path}")
        blob.upload_from_filename(str(local_path), content_type=content_type, timeout=600)
        
        # Generate signed URL
        signed_url = blob.generate_signed_url(
            version="v4",
            expiration=timedelta(days=GCS_URL_EXPIRATION_DAYS),
            method="GET",
            response_type=content_type,
            response_disposition="inline",
        )
        
        gcs_uri = f"gs://{GCS_BUCKET_NAME}/{gcs_path}"
        print(f"✅ Uploaded to GCS: {gcs_uri}")
        
        return True, signed_url, None
        
    except Exception as e:
        error_msg = f"GCS upload failed: {str(e)}"
        print(f"❌ {error_msg}")
        return False, None, error_msg

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL-SPECIFIC CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# LTX-2 settings
LTX2_FPS = 24
LTX2_MIN_FRAMES = 17
LTX2_FRAME_STEP = 8
LTX2_RESOLUTION_DIVISOR = 64

# Wan2.2 settings  
WAN22_FPS = 16
WAN22_MIN_FRAMES = 5
WAN22_FRAME_STEP = 4
WAN22_RESOLUTION_DIVISOR = 16

# Z-Image settings
ZIMAGE_RESOLUTION_DIVISOR = 64

# Model family detection
MODEL_FAMILIES = {
    "ltx2_distilled": "ltx2",
    "ltx2_19B": "ltx2",
    "z_image": "z_image",
    "z_image_control": "z_image",
    "z_image_control2": "z_image",
    "i2v_2_2": "wan22",
    "i2v_2_2_Enhanced_Lightning_v2": "wan22",
    "i2v_2_2_svi2pro": "wan22_svi2pro",
    "i2v_2_2_Enhanced_Lightning_v2_svi2pro": "wan22_svi2pro",
}

# SVI2Pro Sliding Window settings (from env vars or defaults)
SVI2PRO_SLIDING_WINDOW_SIZE = int(os.environ.get("WAN2GP_SLIDING_WINDOW_SIZE", "81"))
SVI2PRO_SLIDING_WINDOW_OVERLAP = int(os.environ.get("WAN2GP_SLIDING_WINDOW_OVERLAP", "4"))
SVI2PRO_COLOR_CORRECTION_STRENGTH = float(os.environ.get("WAN2GP_COLOR_CORRECTION_STRENGTH", "1.0"))
SVI2PRO_TEMPORAL_UPSAMPLING = os.environ.get("WAN2GP_TEMPORAL_UPSAMPLING", "rife2")

# Resolution presets per model family
RESOLUTION_PRESETS = {
    "ltx2": {
        "480p": (832, 480),
        "480p_portrait": (480, 832),
        "720p": (1280, 720),
        "720p_portrait": (720, 1280),
        "768": (768, 768),
        "1024": (1024, 1024),
    },
    "wan22": {
        "default": (832, 480),  # GUI default resolution
        "480p": (832, 480),
        "480p_portrait": (480, 848),
        "480_square": (480, 480),
        "720p": (1280, 720),
        "720p_portrait": (720, 1280),
        "720_square": (720, 720),
        "576p": (1024, 576),
        "576p_portrait": (576, 1024),
        "576_square": (576, 576),
    },
    "wan22_svi2pro": {
        "default": (832, 480),  # GUI default resolution
        "480p": (832, 480),
        "480p_portrait": (480, 848),
        "480_square": (480, 480),
        "720p": (1280, 720),
        "720p_portrait": (720, 1280),
        "720_square": (720, 720),
        "576p": (1024, 576),
        "576p_portrait": (576, 1024),
        "576_square": (576, 576),
    },
    "z_image": {
        "512": (512, 512),
        "768": (768, 768),
        "1024": (1024, 1024),
        "landscape": (1024, 768),
        "portrait": (768, 1024),
        "wide": (1280, 768),
    },
}

# Global model reference
model_instance = None
model_handler = None
model_def = None
current_model_type = None
current_base_model_type = None
current_model_family = None
offloadobj = None
loras_loaded = False  # Track if LoRAs have been loaded

# ═══════════════════════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class ZImageRequest(BaseModel):
    """Request model for Z-Image text-to-image generation"""
    prompt: str = Field(..., description="Text prompt describing the image")
    negative_prompt: str = Field("", description="Negative prompt (not used in turbo mode)")
    
    resolution_preset: Optional[str] = Field(
        "1024",
        description="Resolution preset: 512, 768, 1024, landscape, portrait, wide"
    )
    width: Optional[int] = Field(None, description="Image width (must be multiple of 64)")
    height: Optional[int] = Field(None, description="Image height (must be multiple of 64)")
    
    num_inference_steps: int = Field(8, ge=4, le=20, description="Number of denoising steps (8 for turbo)")
    guidance_scale: float = Field(0.0, ge=0.0, le=10.0, description="CFG scale (0 for turbo)")
    seed: int = Field(-1, description="Random seed (-1 for random)")
    batch_size: int = Field(1, ge=1, le=4, description="Number of images to generate")
    
    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "A majestic lion in a savanna at sunset, photorealistic, 8k",
                "resolution_preset": "1024",
                "num_inference_steps": 8,
                "seed": -1
            }
        }


class LTX2ImageToVideoRequest(BaseModel):
    """Request model for LTX-2 image-to-video generation"""
    prompt: str = Field(..., description="Text prompt describing the video motion/action")
    image_url: str = Field(..., description="URL of the input image to animate")
    duration: float = Field(5.0, ge=0.7, le=20.0, description="Video duration in seconds (0.7-20)")
    
    resolution_preset: Optional[str] = Field(
        None, 
        description="Resolution preset: 480p, 720p, 768, 1024"
    )
    width: Optional[int] = Field(None, description="Video width (must be multiple of 64)")
    height: Optional[int] = Field(None, description="Video height (must be multiple of 64)")
    
    guidance_scale: float = Field(4.0, ge=1.0, le=10.0, description="Classifier-free guidance scale")
    input_video_strength: float = Field(1.0, ge=0.0, le=1.0, description="Image/source video strength (lower values = more motion, 1.0 = faithful to input)")
    num_inference_steps: int = Field(40, ge=8, le=100, description="Number of inference steps (40 for dev model with LoRA)")
    seed: int = Field(-1, description="Random seed (-1 for random)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "A person slowly turning their head and smiling",
                "image_url": "https://example.com/portrait.jpg",
                "duration": 5.0,
                "resolution_preset": "720p",
                "guidance_scale": 4.0,
                "input_video_strength": 1.0,
                "num_inference_steps": 40,
                "seed": -1
            }
        }


class Wan22ImageToVideoRequest(BaseModel):
    """Request model for Wan2.2 I2V Lightning v2 generation"""
    prompt: str = Field(..., description="Text prompt describing the video (supports temporal markers)")
    image_url: str = Field(..., description="URL of the input image to animate")
    duration: float = Field(5.0, ge=0.3, le=30.0, description="Video duration in seconds (up to 30s with sliding window)")
    
    resolution_preset: Optional[Literal["default", "480p", "480p_portrait", "480_square", "720p", "720p_portrait", "720_square", "576p", "576p_portrait", "576_square"]] = Field(
        "default",
        description="Resolution preset: default (832x480), 480p, 720p, 576p (portrait and square variants)"
    )
    width: Optional[int] = Field(None, description="Video width (must be multiple of 16)")
    height: Optional[int] = Field(None, description="Video height (must be multiple of 16)")
    
    # Enhanced Lightning v2 defaults from i2v_2_2_Enhanced_Lightning_v2.json
    num_inference_steps: int = Field(4, ge=4, le=30, description="Inference steps (4 for Lightning v2)")
    guidance_scale: float = Field(1.0, ge=1.0, le=10.0, description="CFG scale phase 1 (1.0 for Lightning v2)")
    guidance2_scale: float = Field(1.0, ge=1.0, le=10.0, description="CFG scale phase 2 (1.0 for Lightning v2)")
    guidance_phases: int = Field(2, ge=1, le=3, description="Number of guidance phases (2 for Lightning v2)")
    model_switch_phase: int = Field(1, ge=1, le=2, description="Phase to switch models (1 for Lightning v2)")
    switch_threshold: int = Field(900, ge=0, le=1000, description="Step threshold to switch phases (900 for Lightning v2)")
    flow_shift: float = Field(5.0, ge=1.0, le=15.0, description="Flow shift parameter (5.0 for Lightning v2)")
    seed: int = Field(-1, description="Random seed (-1 for random)")
    
    # Optional sliding window parameters (auto-enabled for duration > 5s)
    use_sliding_window: Optional[bool] = Field(None, description="Enable sliding window for long videos (auto-enabled if duration > 5s)")
    sliding_window_size: int = Field(81, ge=33, le=257, description="Frames per sliding window (81 default)")
    sliding_window_overlap: int = Field(4, ge=1, le=16, description="Overlap frames between windows")
    sliding_window_overlap_noise: int = Field(20, ge=0, le=100, description="Noise added to overlapped frames to reduce stitching glitch (20 recommended for Lightning)")
    color_correction_strength: float = Field(1.0, ge=0.0, le=1.0, description="Color correction between windows")
    temporal_upsampling: Optional[Literal["", "rife2", "rife4"]] = Field("rife2", description="RIFE temporal upsampling: '' (none), 'rife2' (2x fps), 'rife4' (4x fps)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "(at 0 seconds: wide shot of a woman standing, cinematic lighting). (at 2 seconds: camera slowly zooms in). (at 4 seconds: close-up on face, she smiles).",
                "image_url": "https://example.com/portrait.jpg",
                "duration": 5.0,
                "resolution_preset": "default",
                "temporal_upsampling": "rife2",
                "seed": -1
            }
        }


class SVI2ProImageToVideoRequest(BaseModel):
    """
    Request model for WAN 2.2 SVI2Pro Enhanced Lightning v2 generation.
    
    SVI2Pro (Stable Video Infinity Pro 2) supports potentially unlimited video length
    via the sliding window method. Perfect for videos longer than 10 seconds.
    
    Features:
    - Sliding window generation for long videos
    - RIFE x2 temporal upsampling for smoother output
    - Color correction between windows for consistency
    - 8-step distilled inference
    """
    prompt: str = Field(..., description="Text prompt describing the video (supports temporal markers)")
    image_url: str = Field(..., description="URL of the input image to animate")
    duration: float = Field(10.0, ge=0.3, le=120.0, description="Video duration in seconds (supports long videos via sliding window)")
    
    resolution_preset: Optional[Literal["default", "480p", "480p_portrait", "480_square", "720p", "720p_portrait", "720_square", "576p", "576p_portrait", "576_square"]] = Field(
        "default",
        description="Resolution preset: default (832x480), 480p, 720p, 576p (portrait and square variants)"
    )
    width: Optional[int] = Field(None, description="Video width (must be multiple of 16)")
    height: Optional[int] = Field(None, description="Video height (must be multiple of 16)")
    
    # SVI2Pro Enhanced Lightning v2 defaults (from i2v_2_2_Enhanced_Lightning_v2_svi2pro.json)
    num_inference_steps: int = Field(8, ge=4, le=30, description="Inference steps (8 for SVI2Pro Lightning)")
    guidance_scale: float = Field(1.0, ge=1.0, le=10.0, description="CFG scale phase 1 (1.0 for Lightning)")
    guidance2_scale: float = Field(1.0, ge=1.0, le=10.0, description="CFG scale phase 2 (1.0 for Lightning)")
    guidance_phases: int = Field(2, ge=1, le=3, description="Number of guidance phases (2 for Lightning)")
    model_switch_phase: int = Field(1, ge=1, le=2, description="Phase to switch models (1 for Lightning)")
    switch_threshold: int = Field(900, ge=0, le=1000, description="Step threshold to switch phases (900 for Lightning)")
    flow_shift: float = Field(5.0, ge=1.0, le=15.0, description="Flow shift parameter (5.0 for Lightning)")
    seed: int = Field(-1, description="Random seed (-1 for random)")
    
    # Sliding window settings
    sliding_window_size: int = Field(81, ge=33, le=257, description="Frames per sliding window (81 default)")
    sliding_window_overlap: int = Field(4, ge=1, le=16, description="Overlap frames between windows (4 for SVI2Pro)")
    sliding_window_overlap_noise: int = Field(20, ge=0, le=100, description="Noise added to overlapped frames to reduce stitching glitch (20 recommended)")
    color_correction_strength: float = Field(1.0, ge=0.0, le=1.0, description="Color correction between windows (1.0 recommended)")
    
    # Post-processing
    temporal_upsampling: Optional[Literal["", "rife2", "rife4"]] = Field(
        "rife2",
        description="RIFE temporal upsampling: '' (none), 'rife2' (2x fps), 'rife4' (4x fps)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "(at 0 seconds: a man opens his fridge, takes a beer and drinks it)",
                "image_url": "https://example.com/kitchen.jpg",
                "duration": 10.0,
                "resolution_preset": "default",
                "sliding_window_size": 81,
                "sliding_window_overlap": 4,
                "color_correction_strength": 1.0,
                "temporal_upsampling": "rife2",
                "seed": -1
            }
        }


class ImageToVideoRequest(BaseModel):
    """Generic request model for image-to-video (base64 input)"""
    prompt: str = Field(..., description="Text prompt describing the video")
    image_base64: str = Field(..., description="Base64 encoded start image (PNG/JPEG)")
    negative_prompt: str = Field("", description="Negative prompt")
    duration: float = Field(5.0, ge=0.7, le=20.0, description="Video duration in seconds")
    width: int = Field(768, description="Video width")
    height: int = Field(512, description="Video height")
    num_inference_steps: int = Field(8, description="Number of denoising steps")
    guidance_scale: float = Field(4.0, description="Classifier-free guidance scale")
    seed: int = Field(-1, description="Random seed (-1 for random)")


class TextToVideoRequest(BaseModel):
    """Request model for text-to-video generation"""
    prompt: str = Field(..., description="Text prompt describing the video")
    negative_prompt: str = Field("", description="Negative prompt")
    duration: float = Field(5.0, ge=0.7, le=20.0, description="Video duration in seconds")
    width: int = Field(768, description="Video width")
    height: int = Field(512, description="Video height")
    num_inference_steps: int = Field(8, description="Number of denoising steps")
    guidance_scale: float = Field(4.0, description="Classifier-free guidance scale")
    seed: int = Field(-1, description="Random seed (-1 for random)")


class GenerationResponse(BaseModel):
    """Response model for generation requests"""
    status: str
    job_id: str
    message: Optional[str] = None
    output_url: Optional[str] = None
    video_url: Optional[str] = None  # Alias for video responses
    image_url: Optional[str] = None  # Alias for image responses
    generation_time_seconds: Optional[float] = None
    duration_seconds: Optional[float] = None
    num_frames: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    seed: Optional[int] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_type: Optional[str] = None
    model_family: Optional[str] = None
    gpu_name: Optional[str] = None
    gpu_memory_total_mb: Optional[int] = None
    gpu_memory_used_mb: Optional[int] = None


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_model_family(model_type: str) -> str:
    """Get the model family for a given model type"""
    return MODEL_FAMILIES.get(model_type, "unknown")


def duration_to_frames_ltx2(duration_seconds: float) -> int:
    """Convert duration to valid frame count for LTX-2 (17 + 8*n)"""
    target_frames = int(duration_seconds * LTX2_FPS)
    if target_frames < LTX2_MIN_FRAMES:
        return LTX2_MIN_FRAMES
    n = max(0, (target_frames - LTX2_MIN_FRAMES) // LTX2_FRAME_STEP)
    valid_frames = LTX2_MIN_FRAMES + (n * LTX2_FRAME_STEP)
    next_valid = valid_frames + LTX2_FRAME_STEP
    if abs(next_valid - target_frames) < abs(valid_frames - target_frames):
        valid_frames = next_valid
    return valid_frames


def duration_to_frames_wan22(duration_seconds: float) -> int:
    """Convert duration to valid frame count for Wan2.2 (5 + 4*n)"""
    target_frames = int(duration_seconds * WAN22_FPS)
    if target_frames < WAN22_MIN_FRAMES:
        return WAN22_MIN_FRAMES
    n = max(0, (target_frames - WAN22_MIN_FRAMES) // WAN22_FRAME_STEP)
    valid_frames = WAN22_MIN_FRAMES + (n * WAN22_FRAME_STEP)
    next_valid = valid_frames + WAN22_FRAME_STEP
    if abs(next_valid - target_frames) < abs(valid_frames - target_frames):
        valid_frames = next_valid
    return valid_frames


def frames_to_duration(num_frames: int, fps: int) -> float:
    """Convert frame count to duration in seconds"""
    return round(num_frames / fps, 2)


def estimate_vram_required(width: int, height: int, num_frames: int, model_family: str) -> float:
    """
    Estimate VRAM required for generation in GB.
    This is a rough heuristic based on empirical testing.
    """
    # Base model memory (varies by model)
    base_vram = {
        "ltx2": 12.0,  # LTX-2 base model
        "wan22": 14.0,  # Wan2.2 dual transformer
        "wan22_svi2pro": 14.0,
        "z_image": 8.0,
    }.get(model_family, 12.0)
    
    # Latent space multiplier (models work in compressed latent space)
    # LTX2: 8x spatial, 8x temporal compression
    # Wan2.2: 8x spatial, 4x temporal compression
    if model_family == "ltx2":
        latent_frames = num_frames // 8
        latent_h, latent_w = height // 8, width // 8
    else:
        latent_frames = num_frames // 4
        latent_h, latent_w = height // 8, width // 8
    
    # Rough estimate: latents + attention maps + gradients
    # This is very approximate - actual usage depends on many factors
    latent_size_gb = (latent_w * latent_h * latent_frames * 16 * 4) / (1024**3)
    
    return base_vram + latent_size_gb * 3  # 3x for activations/gradients


def get_gpu_memory_gb() -> tuple[float, float]:
    """Get total and free GPU memory in GB"""
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        total = props.total_memory / (1024**3)
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        return total, total - allocated
    return 0.0, 0.0


def validate_generation_request(
    width: int, height: int, num_frames: int, model_family: str
) -> tuple[bool, str]:
    """
    Validate if a generation request is likely to succeed.
    Returns (is_valid, warning_message).
    """
    total_vram, free_vram = get_gpu_memory_gb()
    estimated_vram = estimate_vram_required(width, height, num_frames, model_family)
    
    # Add 20% safety margin
    if estimated_vram > free_vram * 0.8:
        return False, (
            f"Request may OOM: estimated {estimated_vram:.1f}GB needed, "
            f"only {free_vram:.1f}GB free (of {total_vram:.1f}GB total). "
            f"Try reducing resolution or duration."
        )
    
    return True, ""


def resolve_resolution(
    model_family: str,
    resolution_preset: Optional[str] = None,
    width: Optional[int] = None, 
    height: Optional[int] = None,
) -> tuple[int, int]:
    """Resolve resolution from preset or explicit values for a model family"""
    presets = RESOLUTION_PRESETS.get(model_family, RESOLUTION_PRESETS["ltx2"])
    
    # Use preset if provided
    if resolution_preset and resolution_preset in presets:
        return presets[resolution_preset]
    
    # Get divisor for model family
    if model_family == "wan22":
        divisor = WAN22_RESOLUTION_DIVISOR
        default_w, default_h = 848, 480
    elif model_family == "z_image":
        divisor = ZIMAGE_RESOLUTION_DIVISOR
        default_w, default_h = 1024, 1024
    else:  # ltx2
        divisor = LTX2_RESOLUTION_DIVISOR
        default_w, default_h = 768, 512
    
    # Use explicit values or defaults
    w = width if width is not None else default_w
    h = height if height is not None else default_h
    
    # Align to divisor
    w = (w // divisor) * divisor
    h = (h // divisor) * divisor
    
    # Ensure minimum size
    w = max(256, w)
    h = max(256, h)
    
    return w, h


async def fetch_image_from_url(url: str, timeout: float = 30.0) -> Image.Image:
    """Fetch an image from a URL and return as PIL Image"""
    # Add headers to avoid being blocked by CDNs (Discord, etc.)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "image/*,*/*;q=0.8",
    }
    
    try:
        async with httpx.AsyncClient(
            timeout=timeout, 
            follow_redirects=True,
            verify=True,  # Enable SSL verification
        ) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            image_bytes = response.content
            
            if len(image_bytes) == 0:
                raise ValueError("Received empty response from image URL")
            
            image = Image.open(io.BytesIO(image_bytes))
            return image.convert("RGB")
            
    except httpx.TimeoutException:
        raise RuntimeError(f"Timeout fetching image (>{timeout}s)")
    except httpx.ConnectError as e:
        raise RuntimeError(f"Connection error: {type(e).__name__} - check if URL is accessible")
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"HTTP {e.response.status_code}: {e.response.reason_phrase}")
    except Exception as e:
        # Get exception type name if str(e) is empty
        error_msg = str(e) if str(e) else f"{type(e).__name__} (no details)"
        raise RuntimeError(f"{error_msg}")


def decode_base64_image(image_base64: str) -> Image.Image:
    """Decode a base64 string to PIL Image"""
    if "," in image_base64:
        image_base64 = image_base64.split(",")[1]
    image_bytes = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image_bytes))
    return image.convert("RGB")


def resize_image_preserve_aspect(
    image: Image.Image, 
    target_width: int, 
    target_height: int, 
    block_size: int = 16,
    fit_mode: str = "cover"
) -> Image.Image:
    """
    Resize image with aspect ratio handling similar to GUI's calculate_dimensions_and_resize_image.
    
    Args:
        image: Input PIL image
        target_width: Target width
        target_height: Target height
        block_size: Alignment block size (default 16 for WAN VAE)
        fit_mode: "cover" (crop to fill), "contain" (letterbox), or "stretch" (simple resize)
    
    Returns:
        Resized PIL image with dimensions aligned to block_size
    """
    img_width, img_height = image.size
    
    if fit_mode == "stretch":
        # Simple resize (may distort)
        return image.resize((target_width, target_height), Image.LANCZOS)
    
    elif fit_mode == "cover":
        # Scale and crop to fill target (no letterboxing)
        img_aspect = img_width / img_height
        target_aspect = target_width / target_height
        
        if img_aspect > target_aspect:
            # Image is wider - fit by height, crop width
            new_height = target_height
            new_width = int(target_height * img_aspect)
        else:
            # Image is taller - fit by width, crop height
            new_width = target_width
            new_height = int(target_width / img_aspect)
        
        # Align to block size
        new_width = (new_width // block_size) * block_size
        new_height = (new_height // block_size) * block_size
        
        # Resize
        resized = image.resize((new_width, new_height), Image.LANCZOS)
        
        # Center crop to target
        left = (new_width - target_width) // 2
        top = (new_height - target_height) // 2
        cropped = resized.crop((left, top, left + target_width, top + target_height))
        return cropped
    
    elif fit_mode == "contain":
        # Scale to fit within target, add black letterbox/pillarbox
        img_aspect = img_width / img_height
        target_aspect = target_width / target_height
        
        if img_aspect > target_aspect:
            # Image is wider - fit by width
            new_width = target_width
            new_height = int(target_width / img_aspect)
        else:
            # Image is taller - fit by height
            new_height = target_height
            new_width = int(target_height * img_aspect)
        
        # Align to block size
        new_width = (new_width // block_size) * block_size
        new_height = (new_height // block_size) * block_size
        
        resized = image.resize((new_width, new_height), Image.LANCZOS)
        
        # Create black canvas and paste centered
        canvas = Image.new("RGB", (target_width, target_height), (0, 0, 0))
        left = (target_width - new_width) // 2
        top = (target_height - new_height) // 2
        canvas.paste(resized, (left, top))
        return canvas
    
    else:
        raise ValueError(f"Unknown fit_mode: {fit_mode}")


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_wan2gp_model(model_type: str = DEFAULT_MODEL_TYPE, profile: int = DEFAULT_PROFILE):
    """Load the Wan2GP model into VRAM"""
    global model_instance, model_handler, model_def, current_model_type, current_model_family, current_base_model_type, offloadobj, loras_loaded
    
    print(f"⏳ Loading model: {model_type} (profile: {profile})...")
    start_time = time.time()
    
    from wgp import (
        load_models, 
        get_model_def, 
        get_base_model_type, 
        get_model_handler,
    )
    
    model_def = get_model_def(model_type)
    base_model_type = get_base_model_type(model_type)
    model_handler = get_model_handler(base_model_type)
    
    model_instance, offloadobj = load_models(model_type, override_profile=profile)
    current_model_type = model_type
    current_base_model_type = base_model_type  # Store base model type for generate()
    current_model_family = get_model_family(model_type)
    loras_loaded = False  # Reset LoRA state for new model
    
    load_time = time.time() - start_time
    print(f"✅ Model loaded in {load_time:.1f}s (family: {current_model_family}, base: {base_model_type})")
    
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"   GPU: {props.name}")
        print(f"   VRAM: {props.total_memory // 1024 // 1024}MB")
    
    # Ensure model LoRAs are downloaded
    print("⏳ Checking/downloading model LoRAs...")
    try:
        from wgp import get_model_recursive_prop, get_lora_dir, download_file, get_model_def
        import os
        
        # Debug: check what the model definition contains
        mdef = get_model_def(model_type)
        if mdef:
            print(f"   Model def keys: {list(mdef.keys())[:10]}...")
            if "loras" in mdef:
                print(f"   Found loras in model_def: {mdef['loras']}")
        
        model_loras = get_model_recursive_prop(model_type, "loras", return_list=True)
        print(f"   get_model_recursive_prop('loras') returned: {model_loras}")
        
        if not model_loras:
            # Fallback: check model_def directly
            if mdef and "loras" in mdef:
                model_loras = mdef["loras"]
                print(f"   Using loras from model_def directly: {model_loras}")
        
        if model_loras:
            lora_dir = get_lora_dir(model_type)
            os.makedirs(lora_dir, exist_ok=True)
            print(f"   LoRA dir: {lora_dir}")
            for url in model_loras:
                filename = os.path.join(lora_dir, url.split("/")[-1])
                if not os.path.isfile(filename):
                    if url.startswith("http"):
                        print(f"   Downloading LoRA: {os.path.basename(filename)}")
                        download_file(url, filename)
                        print(f"   ✅ Downloaded {os.path.basename(filename)}")
                else:
                    print(f"   LoRA already exists: {os.path.basename(filename)}")
        else:
            print(f"   No LoRAs found in model definition")
    except Exception as e:
        import traceback
        print(f"   ⚠️ LoRA download check failed: {e}")
        traceback.print_exc()
    
    # NOTE: LoRAs are loaded on first generation, not at startup
    # This matches the GUI behavior where LoRAs are loaded after MMGP fully hooks the model
    # Attempting to load LoRAs here causes "unexpected module keys" warnings because
    # MMGP hasn't finished initializing the transformer's state dictionary yet
    print("   ℹ️  LoRAs will be loaded on first generation (matching GUI behavior)")
    
    return model_instance


def unload_model():
    """Release model from memory"""
    global model_instance, offloadobj, loras_loaded
    
    if offloadobj is not None:
        offloadobj.release()
        offloadobj = None
    
    model_instance = None
    loras_loaded = False  # Reset LoRA state
    gc.collect()
    torch.cuda.empty_cache()


def load_and_configure_loras(
    num_inference_steps: int,
    guidance_phases: int = 2,
    model_switch_phase: int = 1,
    force_reload: bool = False,
) -> dict:
    """
    Load LoRAs into the model and return the properly configured loras_slists.
    
    This function does what the GUI does before generation:
    1. Gets transformer LoRAs from model definition
    2. Parses the multiplier strings (e.g., ["1;0", "0;1"]) into phase schedules
    3. Loads the LoRAs into the model using offload.load_loras_into_model()
    
    LoRAs are only loaded once per model load (unless force_reload=True).
    
    Args:
        num_inference_steps: Number of denoising steps
        guidance_phases: Number of guidance phases (1-3)
        model_switch_phase: Phase at which to switch models
        force_reload: Force reload LoRAs even if already loaded
        
    Returns:
        dict with phase1, phase2, phase3, shared lists
    """
    global loras_loaded
    from wgp import get_transformer_loras, get_loras_preprocessor
    
    try:
        # Get LoRA filenames and multipliers from model definition
        transformer_loras_filenames, transformer_loras_multipliers = get_transformer_loras(current_model_type)
        
        if transformer_loras_filenames is None or len(transformer_loras_filenames) == 0:
            print("   No LoRAs configured for this model")
            return {
                "phase1": [], "phase2": [], "phase3": [], "shared": [],
                "model_switch_step": num_inference_steps,
                "model_switch_step2": num_inference_steps,
            }
        
        # Parse the multipliers into phase structure
        # This handles multiplier strings like "1;0" (phase1=1, phase2=0)
        loras_list_mult_choices_nums, loras_slists, errors = parse_loras_multipliers(
            transformer_loras_multipliers,
            len(transformer_loras_filenames),
            num_inference_steps,
            nb_phases=guidance_phases,
            model_switch_phase=model_switch_phase,
        )
        
        if errors:
            print(f"⚠️ Warning parsing LoRA multipliers: {errors}")
            return {
                "phase1": [], "phase2": [], "phase3": [], "shared": [],
                "model_switch_step": num_inference_steps,
                "model_switch_step2": num_inference_steps,
            }
        
        # Only load LoRAs if not already loaded (or force reload)
        if not loras_loaded or force_reload:
            print(f"   LoRAs found: {len(transformer_loras_filenames)}")
            for i, (fname, mult) in enumerate(zip(transformer_loras_filenames, transformer_loras_multipliers)):
                print(f"     LoRA {i}: {Path(fname).name} (mult: {mult})")
            
            print(f"   Phase 1 multipliers: {loras_slists.get('phase1', [])}")
            print(f"   Phase 2 multipliers: {loras_slists.get('phase2', [])}")
            
            # Load LoRAs into the model - use get_trans_lora() like the GUI does!
            # This returns the correct transformer object for LoRA loading
            if hasattr(model_instance, "get_trans_lora"):
                trans_lora, trans2_lora = model_instance.get_trans_lora()
            else:
                trans_lora = model_instance.model if hasattr(model_instance, 'model') else None
                trans2_lora = None
            
            if trans_lora is not None:
                # Use trans_lora for preprocessing target (matching GUI at wgp.py line 5446)
                preprocess_target = trans_lora
                split_linear_modules_map = getattr(preprocess_target, "split_linear_modules_map", None)
                preprocess_sd = get_loras_preprocessor(preprocess_target, current_base_model_type)
                
                print(f"   Loading LoRAs into model (via get_trans_lora)...")
                offload.load_loras_into_model(
                    trans_lora,
                    transformer_loras_filenames,
                    loras_list_mult_choices_nums,
                    activate_all_loras=True,
                    preprocess_sd=preprocess_sd,
                    split_linear_modules_map=split_linear_modules_map,
                )
                
                # Check for LoRA loading errors (matching GUI at wgp.py line 5458)
                lora_errors = getattr(trans_lora, "_loras_errors", [])
                if lora_errors:
                    error_files = [msg for _, msg in lora_errors]
                    print(f"   ⚠️ LoRA loading warnings: {error_files}")
                else:
                    print(f"   ✅ LoRAs loaded successfully")
                
                # Sync LoRAs to second transformer if present (matching GUI at wgp.py line 5463)
                if trans2_lora is not None:
                    offload.sync_models_loras(trans_lora, trans2_lora)
                    print(f"   ✅ LoRAs synced to trans2")
                
                loras_loaded = True
            else:
                print("   ⚠️ Could not find transformer to load LoRAs into")
        
        return loras_slists
        
    except Exception as e:
        import traceback
        print(f"⚠️ Failed to load LoRAs: {e}")
        traceback.print_exc()
        return {
            "phase1": [], "phase2": [], "phase3": [], "shared": [],
            "model_switch_step": num_inference_steps,
            "model_switch_step2": num_inference_steps,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# GENERATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def generate_image_internal(
    prompt: str,
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 1024,
    num_inference_steps: int = 8,
    guidance_scale: float = 0.0,
    seed: int = -1,
    batch_size: int = 1,
) -> tuple[str, float, dict]:
    """Generate image using Z-Image model"""
    global model_instance, model_def
    
    if model_instance is None:
        raise RuntimeError("Model not loaded")
    
    # Handle seed
    if seed < 0:
        seed = int(torch.randint(0, 2**32 - 1, (1,)).item())
    
    job_id = str(uuid.uuid4())[:8]
    output_path = OUTPUT_DIR / f"{job_id}.png"
    
    print(f"🖼️ Generating image: {prompt[:50]}...")
    print(f"   Resolution: {width}x{height}, Steps: {num_inference_steps}")
    
    start_time = time.time()
    
    # Set up offload shared state - use detected attention mode
    offload.shared_state["_attention"] = DETECTED_ATTENTION_MODE
    offload.shared_state["_chipmunk"] = False
    offload.shared_state["_radial"] = False
    offload.shared_state["_nag_scale"] = 1.0
    offload.shared_state["_nag_tau"] = 3.5
    offload.shared_state["_nag_alpha"] = 0.5
    
    model_instance._interrupt = False
    
    # Progress callback (required by some models like Z-Image)
    def progress_callback(step, latents=None, force_update=False, total_steps=None, **kwargs):
        if step >= 0:
            steps_display = total_steps if total_steps else num_inference_steps
            print(f"   Step {step + 1}/{steps_display}")
    
    try:
        result = model_instance.generate(
            input_prompt=prompt,
            n_prompt=negative_prompt if negative_prompt else None,
            width=width,
            height=height,
            sampling_steps=num_inference_steps,
            guide_scale=guidance_scale,
            seed=seed,
            callback=progress_callback,
            # loras_slists: NOT passed - let model use its internal LoRA configuration
        )
        
        # Save image
        if isinstance(result, dict):
            image_tensor = result.get("x", result)
        else:
            image_tensor = result
        
        # Convert tensor to PIL and save
        if hasattr(image_tensor, 'cpu'):
            # Convert bf16 -> float32 before numpy (numpy doesn't support bf16)
            img_tensor = image_tensor.squeeze().float().cpu()
            
            # Handle channel dimension (C, H, W) -> (H, W, C)
            if img_tensor.dim() == 3 and img_tensor.shape[0] in (1, 3, 4):
                img_tensor = img_tensor.permute(1, 2, 0)
            
            # Handle grayscale
            if img_tensor.dim() == 3 and img_tensor.shape[-1] == 1:
                img_tensor = img_tensor.squeeze(-1)
            
            img_np = img_tensor.numpy()
            
            # Z-Image outputs in range [-1, 1], need to convert to [0, 255]
            # Check if values are in negative range (indicates -1 to 1 normalization)
            if img_np.min() < 0:
                # Range is [-1, 1] -> convert to [0, 1] first
                img_np = (img_np + 1.0) / 2.0
            
            # Now convert [0, 1] to [0, 255]
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
            else:
                img_np = img_np.clip(0, 255).astype(np.uint8)
            
            Image.fromarray(img_np).save(str(output_path))
        elif isinstance(image_tensor, Image.Image):
            image_tensor.save(str(output_path))
        else:
            raise RuntimeError(f"Unknown image format: {type(image_tensor)}")
            
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"Generation failed: {str(e)}")
    finally:
        gc.collect()
        torch.cuda.empty_cache()
    
    generation_time = time.time() - start_time
    print(f"✅ Image saved to {output_path} in {generation_time:.1f}s")
    
    metadata = {
        "width": width,
        "height": height,
        "seed": seed,
    }
    
    return str(output_path), generation_time, metadata


def convert_pil_image_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert PIL Image to tensor in range [-1, 1] with shape (C, H, W)"""
    return torch.from_numpy(np.array(image).astype(np.float32)).div_(127.5).sub_(1.).movedim(-1, 0)


def generate_video_internal(
    prompt: str,
    image_start: Optional[Image.Image] = None,
    negative_prompt: str = "",
    width: int = 768,
    height: int = 512,
    num_frames: int = 121,
    num_inference_steps: int = 8,
    guidance_scale: float = 4.0,
    guidance2_scale: Optional[float] = None,
    guidance_phases: int = 1,
    model_switch_phase: int = 1,
    switch_threshold: int = 0,
    flow_shift: Optional[float] = None,
    input_video_strength: float = 1.0,
    seed: int = -1,
    fps: int = 24,
) -> tuple[str, float, dict]:
    """Generate video using LTX-2 or Wan2.2 model"""
    global model_instance, model_def, current_model_family
    
    if model_instance is None:
        raise RuntimeError("Model not loaded")
    
    # Handle seed
    if seed < 0:
        seed = int(torch.randint(0, 2**32 - 1, (1,)).item())
    
    job_id = str(uuid.uuid4())[:8]
    output_path = OUTPUT_DIR / f"{job_id}.mp4"
    
    print(f"🎬 Generating video: {prompt[:50]}...")
    print(f"   Resolution: {width}x{height}, Frames: {num_frames}, Steps: {num_inference_steps}")
    print(f"   Duration: {frames_to_duration(num_frames, fps)}s @ {fps}fps")
    print(f"   Attention mode: {DETECTED_ATTENTION_MODE}")
    if image_start is not None:
        print(f"   Input image: {image_start.size[0]}x{image_start.size[1]}")
    
    start_time = time.time()
    timing_log = []
    
    # Set up offload shared state - use detected attention mode
    offload.shared_state["_attention"] = DETECTED_ATTENTION_MODE
    offload.shared_state["_chipmunk"] = False
    offload.shared_state["_radial"] = False
    offload.shared_state["_nag_scale"] = 1.0
    offload.shared_state["_nag_tau"] = 3.5
    offload.shared_state["_nag_alpha"] = 0.5
    
    model_instance._interrupt = False
    
    # Load LoRAs and generate proper loras_slists from model's LoRA configuration
    # For Lightning and SVI2Pro models, this parses multipliers into phase schedules
    # AND loads the LoRAs into the model (critical step the GUI does!)
    t0 = time.time()
    loras_slists = load_and_configure_loras(
        num_inference_steps=num_inference_steps,
        guidance_phases=guidance_phases,
        model_switch_phase=model_switch_phase,
    )
    timing_log.append(f"LoRA config: {time.time()-t0:.1f}s")
    
    # Progress callback (required by Wan models)
    def video_progress_callback(step, latents=None, force_update=False, override_num_inference_steps=None, denoising_extra="", **kwargs):
        if step >= 0:
            steps_display = override_num_inference_steps if override_num_inference_steps else num_inference_steps
            extra = f" {denoising_extra}" if denoising_extra else ""
            print(f"   Step {step + 1}/{steps_display}{extra}")
    
    # Convert image_start to tensor format for Wan2.2 i2v models
    # Wan2.2 i2v models expect input_video as a tensor with shape (C, T, H, W)
    image_start_tensor = None
    input_video_tensor = None
    pre_video_frame_pil = None  # For SVI2Pro mode
    
    if image_start is not None and current_model_family in ["wan22", "wan22_svi2pro"]:
        # For Wan2.2 i2v: convert PIL image to tensor and pass as input_video
        image_start_tensor = convert_pil_image_to_tensor(image_start)
        # Add time dimension: (C, H, W) -> (C, 1, H, W)
        input_video_tensor = image_start_tensor.unsqueeze(1)
        # Ensure tensor is float type (GUI does this check at line 6037-6038)
        if input_video_tensor.dtype == torch.uint8:
            input_video_tensor = input_video_tensor.float().div_(127.5).sub_(1.0)
        # For SVI2Pro, also keep reference to PIL image
        pre_video_frame_pil = image_start
        print(f"   Converted to input_video tensor: {input_video_tensor.shape}, dtype: {input_video_tensor.dtype}")
    
    # Calculate VAE tile size based on resolution and frame count
    # Enable tiling only for very large generations to avoid OOM
    # Tiling adds overhead so only use when necessary
    vae_tile_size = 0  # Default: no tiling (faster)
    # if num_frames > 300 or (width >= 1280 and num_frames > 200):
    #     vae_tile_size = 256
    #     print(f"   Enabling VAE tiling (tile_size=256) for large generation")
    
    # Build generation kwargs
    gen_kwargs = {
        "input_prompt": prompt,
        "n_prompt": negative_prompt if negative_prompt else None,
        "image_start": image_start_tensor,  # Tensor for Wan2.2, PIL for LTX2
        "image_end": None,
        "width": width,
        "height": height,
        "frame_num": num_frames,
        "sampling_steps": num_inference_steps,
        "guide_scale": guidance_scale,
        "seed": seed,
        "fps": float(fps),
        "VAE_tile_size": vae_tile_size,
        "loras_slists": loras_slists,  # Properly configured LoRA phase multipliers
        "callback": video_progress_callback,
        "input_video_strength": input_video_strength,  # For LTX2: controls motion vs faithfulness to input
    }
    
    # For Wan2.2 i2v models, pass the image as input_video (required for i2v_class flow)
    if input_video_tensor is not None:
        gen_kwargs["input_video"] = input_video_tensor
        
        # For SVI2Pro models, also pass pre_video_frame and window_no
        # This is required by the SVI2Pro logic in any2video.py (lines 647-651)
        if current_model_family == "wan22_svi2pro" and pre_video_frame_pil is not None:
            gen_kwargs["pre_video_frame"] = pre_video_frame_pil  # PIL Image
            gen_kwargs["window_no"] = 1  # First window
            gen_kwargs["prefix_video"] = input_video_tensor  # Same as input_video for first frame
            gen_kwargs["prefix_frames_count"] = 1
            print(f"   SVI2Pro: Set pre_video_frame, window_no=1, prefix_video")
            
    elif image_start is not None and current_model_family not in ["wan22", "wan22_svi2pro"]:
        # For LTX2 and other models, pass PIL image as image_start
        gen_kwargs["image_start"] = image_start
    
    # Add Wan2.2 specific parameters
    # NOTE: model.generate() uses "shift" not "flow_shift"
    if flow_shift is not None:
        gen_kwargs["shift"] = flow_shift
    
    # Add dual-phase guidance parameters for Wan2.2
    # NOTE: model.generate() uses "guide_phases" not "guidance_phases", and "guide2_scale" not "guidance2_scale"
    if guidance_phases >= 1:
        gen_kwargs["guide_phases"] = guidance_phases
        gen_kwargs["model_switch_phase"] = model_switch_phase
        gen_kwargs["switch_threshold"] = switch_threshold
        if guidance2_scale is not None:
            gen_kwargs["guide2_scale"] = guidance2_scale
    
    # For WAN22 models, use unipc solver for better consistency
    if current_model_family in ["wan22", "wan22_svi2pro"]:
        gen_kwargs["sample_solver"] = "unipc"
        
        # Additional WAN22 parameters to match GUI (lines 6091-6127 in wgp.py)
        gen_kwargs["causal_block_size"] = 5
        gen_kwargs["causal_attention"] = True
        
        # NAG (Normalized Attention Guidance) parameters - GUI passes these (lines 6109-6111)
        # NAG_scale=1 means disabled, >1 enables negative prompt enforcement even without CFG
        gen_kwargs["NAG_scale"] = 1.0  # Disabled by default
        gen_kwargs["NAG_tau"] = 3.5
        gen_kwargs["NAG_alpha"] = 0.5
        
        # Video prompt type for special modes - empty string for standard i2v
        gen_kwargs["video_prompt_type"] = ""
        
        # Window start frame for multi-window generation
        gen_kwargs["window_start_frame_no"] = 0  # First window starts at 0
        
        # Image mode: 0 = video output, 1 = image output
        gen_kwargs["image_mode"] = 0
    
    # CRITICAL: Pass model_type and offloadobj - required for the model to function
    gen_kwargs["model_type"] = current_base_model_type
    gen_kwargs["offloadobj"] = offloadobj
    
    # Provide a no-op set_header_text callback (used for UI status updates)
    gen_kwargs["set_header_text"] = lambda txt: print(f"   Phase: {txt}")
    
    try:
        # Initialize cache attribute (required by any2video.py for step-skipping logic)
        # Set to None to disable step-skipping cache (TeaCache/MagCache)
        # The transformer model accesses self.cache in forward(), so we must set it
        # For dual-phase models, there are TWO transformers (model and model2)
        if hasattr(model_instance, 'model') and model_instance.model is not None:
            # Use object.__setattr__ to bypass PyTorch module's __setattr__
            object.__setattr__(model_instance.model, 'cache', None)
            print(f"   Set model.cache = None")
        if hasattr(model_instance, 'model2') and model_instance.model2 is not None:
            object.__setattr__(model_instance.model2, 'cache', None)
            print(f"   Set model2.cache = None")
        
        t_gen = time.time()
        result = model_instance.generate(**gen_kwargs)
        timing_log.append(f"model.generate(): {time.time()-t_gen:.1f}s")
        
        # Extract video tensor and audio
        t_post = time.time()
        if isinstance(result, dict):
            video_tensor = result.get("x", result)
            audio_data = result.get("audio", None)
            audio_sr = result.get("audio_sampling_rate", 48000)
        else:
            video_tensor = result
            audio_data = None
            audio_sr = 48000
        
        # Save video (without audio first)
        save_video(video_tensor, str(output_path), fps=fps)
        timing_log.append(f"save_video: {time.time()-t_post:.1f}s")
        
        # Mux audio if present
        if audio_data is not None:
            try:
                # Convert numpy array to torch tensor if needed
                if isinstance(audio_data, np.ndarray):
                    audio_tensor = torch.from_numpy(audio_data)
                else:
                    audio_tensor = audio_data
                # Ensure audio is in (channels, samples) format for torchaudio
                # LTX2 returns (samples, channels) so we need to transpose
                if audio_tensor.dim() == 1:
                    # Mono audio without channel dim -> (1, samples)
                    audio_tensor = audio_tensor.unsqueeze(0)
                elif audio_tensor.dim() == 2:
                    # If shape is (samples, channels) where samples >> channels, transpose to (channels, samples)
                    if audio_tensor.shape[0] > audio_tensor.shape[1] and audio_tensor.shape[1] in (1, 2):
                        audio_tensor = audio_tensor.T
                
                # Try using ffmpeg directly first (more reliable than torchaudio)
                temp_video_path = output_path.with_name(output_path.stem + '_tmp.mp4')
                temp_audio_path = output_path.with_name(output_path.stem + '_tmp.wav')
                output_path.rename(temp_video_path)
                
                try:
                    # Save audio to temp WAV file (avoid torchcodec dependency)
                    audio_np = audio_tensor.cpu().float().numpy()
                    # Normalize to [-1, 1] range if needed
                    if audio_np.max() > 1.0 or audio_np.min() < -1.0:
                        audio_np = audio_np / max(abs(audio_np.max()), abs(audio_np.min()))
                    
                    # Try scipy first, then soundfile
                    try:
                        import scipy.io.wavfile as wavfile
                        # scipy expects (samples, channels) for stereo, (samples,) for mono
                        # and int16 format
                        if audio_np.ndim == 2:
                            audio_np = audio_np.T  # (channels, samples) -> (samples, channels)
                        audio_int16 = (audio_np * 32767).astype(np.int16)
                        wavfile.write(str(temp_audio_path), audio_sr, audio_int16)
                    except ImportError:
                        import soundfile as sf
                        # soundfile expects (samples, channels) for stereo
                        if audio_np.ndim == 2:
                            audio_np = audio_np.T
                        sf.write(str(temp_audio_path), audio_np, audio_sr)
                    
                    # Use ffmpeg to mux video and audio
                    import subprocess
                    cmd = [
                        "ffmpeg", "-y",
                        "-i", str(temp_video_path),
                        "-i", str(temp_audio_path),
                        "-c:v", "copy",
                        "-c:a", "aac",
                        "-b:a", "192k",
                        "-shortest",
                        str(output_path)
                    ]
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                    if result.returncode != 0:
                        raise RuntimeError(f"ffmpeg failed: {result.stderr}")
                    print("   ✅ Audio muxed with ffmpeg")
                    
                except Exception as ffmpeg_err:
                    print(f"   ⚠️ ffmpeg muxing failed: {ffmpeg_err}")
                    # Fall back to torchaudio/remux_with_audio
                    from postprocessing.mmaudio.data.av_utils import remux_with_audio
                    remux_with_audio(temp_video_path, output_path, audio_tensor, audio_sr)
                    print("   ✅ Audio muxed with torchaudio")
                
                finally:
                    temp_video_path.unlink(missing_ok=True)
                    temp_audio_path.unlink(missing_ok=True)
                    
            except Exception as audio_err:
                print(f"   ⚠️ Audio muxing failed, saving video without audio: {audio_err}")
                # If we renamed the file, rename it back
                temp_video_path = output_path.with_name(output_path.stem + '_tmp.mp4')
                if temp_video_path.exists() and not output_path.exists():
                    temp_video_path.rename(output_path)
        
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"Generation failed: {str(e)}")
    finally:
        gc.collect()
        torch.cuda.empty_cache()
    
    generation_time = time.time() - start_time
    print(f"✅ Video saved to {output_path} in {generation_time:.1f}s")
    print(f"   Timing breakdown: {' | '.join(timing_log)}")
    
    metadata = {
        "num_frames": num_frames,
        "duration": frames_to_duration(num_frames, fps),
        "fps": fps,
        "width": width,
        "height": height,
        "seed": seed,
    }
    
    return str(output_path), generation_time, metadata


def download_flownet_if_needed() -> str:
    """
    Download flownet.pkl (RIFE model) if not present.
    Returns the path to flownet.pkl.
    """
    from shared.utils import files_locator as fl
    from huggingface_hub import hf_hub_download
    
    # Check if already exists
    flownet_path = fl.locate_file("flownet.pkl", error_if_none=False)
    if flownet_path is not None:
        return flownet_path
    
    # Download from HuggingFace
    print("📥 Downloading flownet.pkl (RIFE model)...")
    download_dir = fl.get_download_location()
    
    hf_hub_download(
        repo_id="DeepBeepMeep/Wan2.1",
        filename="flownet.pkl",
        local_dir=download_dir
    )
    
    # Verify download
    flownet_path = fl.locate_file("flownet.pkl", error_if_none=True)
    print(f"✅ Downloaded flownet.pkl to {flownet_path}")
    return flownet_path


def perform_rife_upsampling(sample: torch.Tensor, temporal_upsampling: str, fps: int) -> tuple:
    """
    Apply RIFE temporal upsampling to video tensor.
    
    Args:
        sample: Video tensor (C, T, H, W)
        temporal_upsampling: 'rife2' for 2x, 'rife4' for 4x
        fps: Original fps
        
    Returns:
        tuple: (upsampled_tensor, output_fps)
    """
    if temporal_upsampling not in ["rife2", "rife4"]:
        return sample, fps
    
    exp = 1 if temporal_upsampling == "rife2" else 2
    output_fps = fps * (2 ** exp)
    
    print(f"🎬 Applying RIFE temporal upsampling ({temporal_upsampling})...")
    print(f"   Input: {sample.shape[1]} frames @ {fps} fps")
    
    try:
        from postprocessing.rife.inference import temporal_interpolation
        
        # Download flownet.pkl if needed, then get path
        rife_model_path = download_flownet_if_needed()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # temporal_interpolation(model_path, frames, exp, device)
        sample = temporal_interpolation(rife_model_path, sample, exp, device)
        
        print(f"   Output: {sample.shape[1]} frames @ {output_fps} fps")
        
    except Exception as e:
        print(f"⚠️ RIFE upsampling failed: {e}, returning original")
        import traceback
        traceback.print_exc()
        return sample, fps
    
    return sample, output_fps


def generate_video_sliding_window_internal(
    prompt: str,
    image_start: Optional[Image.Image] = None,
    negative_prompt: str = "",
    width: int = 832,
    height: int = 480,
    num_frames: int = 81,
    num_inference_steps: int = 8,
    guidance_scale: float = 1.0,
    guidance2_scale: float = 1.0,
    guidance_phases: int = 2,
    model_switch_phase: int = 1,
    switch_threshold: int = 900,
    flow_shift: float = 5.0,
    seed: int = -1,
    fps: int = 16,
    sliding_window_size: int = 81,
    sliding_window_overlap: int = 4,
    sliding_window_overlap_noise: int = 20,
    color_correction_strength: float = 1.0,
    temporal_upsampling: str = "rife2",
) -> tuple[str, float, dict]:
    """
    Generate video using sliding window technique for long videos.
    
    This function implements proper multi-window generation like the GUI:
    - Generates video in chunks of sliding_window_size frames
    - Uses overlapped_latents to maintain continuity between windows
    - Stitches windows together with proper overlap handling
    - Applies RIFE temporal upsampling at the end
    
    The overlap_noise parameter adds noise to overlapped frames to reduce
    visible stitching artifacts/glitches at window boundaries. Default 20
    is recommended for Lightning LoRA models.
    
    Works with both Lightning and SVI2Pro models.
    """
    global model_instance, model_def, current_model_family
    
    if model_instance is None:
        raise RuntimeError("Model not loaded")
    
    # Handle seed
    if seed < 0:
        seed = int(torch.randint(0, 2**32 - 1, (1,)).item())
    
    job_id = str(uuid.uuid4())[:8]
    output_path = OUTPUT_DIR / f"{job_id}.mp4"
    
    # Latent stride (Wan2.2 uses 4x temporal compression)
    latent_size = 4
    
    # Calculate number of windows needed
    # First window generates sliding_window_size frames
    # Each subsequent window adds (sliding_window_size - overlap) new frames
    # GUI: reuse_frames = min(sliding_window_size - latent_size, sliding_window_overlap)
    # sliding_window_overlap is in video frames (typically 4), NOT latent frames
    reuse_frames = min(sliding_window_size - latent_size, sliding_window_overlap)
    frames_per_subsequent_window = sliding_window_size - reuse_frames
    
    if num_frames <= sliding_window_size:
        total_windows = 1
    else:
        remaining_after_first = num_frames - sliding_window_size
        additional_windows = (remaining_after_first + frames_per_subsequent_window - 1) // frames_per_subsequent_window
        total_windows = 1 + additional_windows
    
    print(f"🎬 Generating video with SLIDING WINDOW: {prompt[:50]}...")
    print(f"   Resolution: {width}x{height}, Total Frames: {num_frames}, Steps: {num_inference_steps}")
    print(f"   Duration: {frames_to_duration(num_frames, fps)}s @ {fps}fps")
    print(f"   Sliding Window: size={sliding_window_size}, overlap={sliding_window_overlap}, overlap_noise={sliding_window_overlap_noise} (reuse_frames={reuse_frames})")
    print(f"   Windows needed: {total_windows}")
    print(f"   Color Correction: {color_correction_strength}, Temporal Upsampling: {temporal_upsampling}")
    if image_start is not None:
        print(f"   Input image: {image_start.size[0]}x{image_start.size[1]}")
    
    start_time = time.time()
    
    # Set up offload shared state
    offload.shared_state["_attention"] = DETECTED_ATTENTION_MODE
    offload.shared_state["_chipmunk"] = False
    offload.shared_state["_radial"] = False
    
    model_instance._interrupt = False
    
    # Load LoRAs
    loras_slists = load_and_configure_loras(
        num_inference_steps=num_inference_steps,
        guidance_phases=guidance_phases,
        model_switch_phase=model_switch_phase,
    )
    
    # Convert image_start to tensor
    image_start_tensor = None
    input_video_tensor = None
    pre_video_frame_pil = None
    
    if image_start is not None:
        image_start_tensor = convert_pil_image_to_tensor(image_start)
        input_video_tensor = image_start_tensor.unsqueeze(1)
        if input_video_tensor.dtype == torch.uint8:
            input_video_tensor = input_video_tensor.float().div_(127.5).sub_(1.0)
        pre_video_frame_pil = image_start
        print(f"   image_start_tensor: {image_start_tensor.shape}, input_video: {input_video_tensor.shape}")
    
    # Initialize cache
    if hasattr(model_instance, 'model') and model_instance.model is not None:
        object.__setattr__(model_instance.model, 'cache', None)
    if hasattr(model_instance, 'model2') and model_instance.model2 is not None:
        object.__setattr__(model_instance.model2, 'cache', None)
    
    # Variables for window loop
    all_video_chunks = []
    overlapped_latents = None
    prefix_video = input_video_tensor
    frames_generated = 0
    window_start_frame = 0
    
    try:
        for window_no in range(1, total_windows + 1):
            print(f"\n   ═══ Window {window_no}/{total_windows} ═══")
            
            # Calculate frames for this window
            if window_no == 1:
                current_window_frames = min(sliding_window_size, num_frames)
                prefix_frames_count = 1 if image_start is not None else 0
            else:
                remaining_frames = num_frames - frames_generated
                current_window_frames = min(sliding_window_size, remaining_frames + reuse_frames)
                prefix_frames_count = reuse_frames
            
            # Align to latent size (frames must be 1 + 4*n)
            current_window_frames = ((current_window_frames - 1) // latent_size) * latent_size + 1
            
            print(f"   Generating {current_window_frames} frames (prefix_frames_count={prefix_frames_count})")
            
            # Progress callback for this window
            def make_callback(win_no):
                def video_progress_callback(step, latents=None, force_update=False, override_num_inference_steps=None, denoising_extra="", **kwargs):
                    if step >= 0:
                        steps_display = override_num_inference_steps if override_num_inference_steps else num_inference_steps
                        extra = f" {denoising_extra}" if denoising_extra else ""
                        print(f"   Step {step + 1}/{steps_display}{extra}")
                return video_progress_callback
            
            # Build kwargs for this window
            gen_kwargs = {
                "input_prompt": prompt,
                "n_prompt": negative_prompt if negative_prompt else None,
                "image_start": image_start_tensor if window_no == 1 else None,
                "image_end": None,
                "width": width,
                "height": height,
                "frame_num": current_window_frames,
                "sampling_steps": num_inference_steps,
                "guide_scale": guidance_scale,
                "seed": seed,
                "VAE_tile_size": 0,
                "loras_slists": loras_slists,
                "callback": make_callback(window_no),
                "shift": flow_shift,
                "guide_phases": guidance_phases,
                "model_switch_phase": model_switch_phase,
                "switch_threshold": switch_threshold,
                "guide2_scale": guidance2_scale,
                "sample_solver": "unipc",
                "causal_block_size": 5,
                "causal_attention": True,
                "fps": fps,
                "overlap_size": sliding_window_overlap,
                "overlap_noise": sliding_window_overlap_noise,
                "color_correction_strength": color_correction_strength,
                "NAG_scale": 1.0,
                "NAG_tau": 3.5,
                "NAG_alpha": 0.5,
                "video_prompt_type": "",
                "image_mode": 0,
                "model_type": current_base_model_type,
                "offloadobj": offloadobj,
                "set_header_text": lambda txt: print(f"   Phase: {txt}"),
                "window_no": window_no,
                "window_start_frame_no": window_start_frame,
                "prefix_frames_count": prefix_frames_count,
            }
            
            # Window 1: use input image
            if window_no == 1:
                if input_video_tensor is not None:
                    gen_kwargs["input_video"] = input_video_tensor
                    gen_kwargs["prefix_video"] = input_video_tensor
                    gen_kwargs["pre_video_frame"] = pre_video_frame_pil
            else:
                # Subsequent windows: use overlapped latents from previous window
                if overlapped_latents is not None:
                    gen_kwargs["overlapped_latents"] = overlapped_latents
                if prefix_video is not None:
                    gen_kwargs["prefix_video"] = prefix_video
                    gen_kwargs["input_video"] = prefix_video
                    # Get last frame of prefix_video as pre_video_frame
                    from shared.utils.utils import convert_tensor_to_image
                    if prefix_video.shape[1] > 0:
                        gen_kwargs["pre_video_frame"] = convert_tensor_to_image(prefix_video[:, -1])
            
            # Request latent slice for next window (if not last window)
            if window_no < total_windows:
                # Get the last few latent frames for continuity
                return_latent_slice = slice(-max(1, (reuse_frames) // latent_size), None)
                gen_kwargs["return_latent_slice"] = return_latent_slice
            
            # Generate this window
            result = model_instance.generate(**gen_kwargs)
            
            # Extract video tensor and latent slice
            if isinstance(result, dict):
                video_chunk = result.get("x", result)
                overlapped_latents = result.get("latent_slice", None)
            else:
                video_chunk = result
                overlapped_latents = None
            
            # For next window, save the last reuse_frames as prefix_video
            if window_no < total_windows and video_chunk is not None:
                # video_chunk is (C, T, H, W)
                if video_chunk.shape[1] > reuse_frames:
                    prefix_video = video_chunk[:, -reuse_frames:].clone()
                    print(f"   Saved prefix_video for next window: {prefix_video.shape}")
            
            # Trim overlap from beginning (except for first window)
            if window_no == 1:
                trimmed_chunk = video_chunk
                new_frames = current_window_frames
            else:
                # Remove the overlapped frames from the beginning
                trimmed_chunk = video_chunk[:, reuse_frames:]
                new_frames = trimmed_chunk.shape[1]
            
            all_video_chunks.append(trimmed_chunk)
            frames_generated += new_frames
            window_start_frame += new_frames
            
            print(f"   Window {window_no} complete: {new_frames} new frames, total: {frames_generated}")
            
            # Clear cache between windows
            gc.collect()
            torch.cuda.empty_cache()
        
        # Concatenate all chunks
        print(f"\n   Concatenating {len(all_video_chunks)} video chunks...")
        video_tensor = torch.cat(all_video_chunks, dim=1)
        print(f"   Final video shape: {video_tensor.shape}")
        
        # Apply RIFE temporal upsampling if requested
        output_fps = fps
        if temporal_upsampling and temporal_upsampling in ["rife2", "rife4"]:
            video_tensor, output_fps = perform_rife_upsampling(video_tensor, temporal_upsampling, fps)
        
        # Save video
        save_video(video_tensor, str(output_path), fps=output_fps)
        
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"Generation failed: {str(e)}")
    finally:
        gc.collect()
        torch.cuda.empty_cache()
    
    generation_time = time.time() - start_time
    print(f"✅ Sliding window video saved to {output_path} in {generation_time:.1f}s")
    
    metadata = {
        "num_frames": num_frames,
        "duration": frames_to_duration(num_frames, fps),
        "fps": output_fps,
        "width": width,
        "height": height,
        "seed": seed,
    }
    
    return str(output_path), generation_time, metadata


# Alias for backward compatibility
generate_video_svi2pro_internal = generate_video_sliding_window_internal


# ═══════════════════════════════════════════════════════════════════════════════
# FASTAPI APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan handler for model loading/unloading"""
    print("🚀 Starting Wan2GP Multi-Model API Server...")
    print(f"   Configured model: {DEFAULT_MODEL_TYPE}")
    try:
        load_wan2gp_model(DEFAULT_MODEL_TYPE, DEFAULT_PROFILE)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        traceback.print_exc()
    
    yield
    
    print("🛑 Shutting down...")
    unload_model()


app = FastAPI(
    title="Wan2GP Multi-Model API",
    description="REST API for Z-Image, LTX-2, and Wan2.2 generation",
    version="3.0.0",
    lifespan=lifespan,
)

# Mount outputs directory
app.mount("/download", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS: HEALTH & INFO
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/", response_model=HealthResponse)
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    gpu_name = gpu_memory_total = gpu_memory_used = None
    
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu_name = props.name
        gpu_memory_total = props.total_memory // 1024 // 1024
        gpu_memory_used = torch.cuda.memory_allocated(0) // 1024 // 1024
    
    return HealthResponse(
        status="healthy" if model_instance is not None else "degraded",
        model_loaded=model_instance is not None,
        model_type=current_model_type,
        model_family=current_model_family,
        gpu_name=gpu_name,
        gpu_memory_total_mb=gpu_memory_total,
        gpu_memory_used_mb=gpu_memory_used,
    )


@app.get("/info")
async def get_info():
    """Get API and model information"""
    return {
        "api_version": "3.0.0",
        "model_type": current_model_type,
        "model_family": current_model_family,
        "model_loaded": model_instance is not None,
        "supported_models": {
            "z_image": {
                "type": "text-to-image",
                "resolution_presets": RESOLUTION_PRESETS["z_image"],
            },
            "ltx2": {
                "type": "image-to-video",
                "fps": LTX2_FPS,
                "min_frames": LTX2_MIN_FRAMES,
                "frame_step": LTX2_FRAME_STEP,
                "resolution_presets": RESOLUTION_PRESETS["ltx2"],
            },
            "wan22": {
                "type": "image-to-video",
                "fps": WAN22_FPS,
                "min_frames": WAN22_MIN_FRAMES,
                "frame_step": WAN22_FRAME_STEP,
                "resolution_presets": RESOLUTION_PRESETS["wan22"],
            },
            "wan22_svi2pro": {
                "type": "image-to-video-long",
                "description": "SVI2Pro for long videos with sliding window",
                "fps": WAN22_FPS,
                "output_fps_rife2": WAN22_FPS * 2,
                "min_frames": WAN22_MIN_FRAMES,
                "frame_step": WAN22_FRAME_STEP,
                "resolution_presets": RESOLUTION_PRESETS["wan22_svi2pro"],
                "sliding_window": {
                    "default_size": SVI2PRO_SLIDING_WINDOW_SIZE,
                    "default_overlap": SVI2PRO_SLIDING_WINDOW_OVERLAP,
                    "default_color_correction": SVI2PRO_COLOR_CORRECTION_STRENGTH,
                    "default_temporal_upsampling": SVI2PRO_TEMPORAL_UPSAMPLING,
                },
            },
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS: Z-IMAGE (Text-to-Image)
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/generate/image", response_model=GenerationResponse)
async def generate_image(request: ZImageRequest, http_request: Request):
    """
    Generate an image from a text prompt (Z-Image)
    
    This endpoint uses Z-Image Turbo for fast text-to-image generation.
    Recommended settings: 8 steps, guidance_scale 0.
    """
    if model_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if current_model_family != "z_image":
        raise HTTPException(
            status_code=400, 
            detail=f"Wrong model loaded. Need z_image, got {current_model_family}. Use /reload endpoint."
        )
    
    job_id = str(uuid.uuid4())[:8]
    
    try:
        width, height = resolve_resolution(
            "z_image",
            resolution_preset=request.resolution_preset,
            width=request.width,
            height=request.height,
        )
        
        output_path, gen_time, metadata = generate_image_internal(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=width,
            height=height,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            seed=request.seed,
            batch_size=request.batch_size,
        )
        
        filename = Path(output_path).name
        base_url = str(http_request.base_url).rstrip("/")
        full_url = f"{base_url}/download/{filename}"
        
        return GenerationResponse(
            status="success",
            job_id=filename.replace(".png", ""),
            output_url=full_url,
            image_url=full_url,
            generation_time_seconds=round(gen_time, 2),
            width=metadata["width"],
            height=metadata["height"],
            seed=metadata["seed"],
        )
        
    except Exception as e:
        traceback.print_exc()
        return GenerationResponse(status="error", job_id=job_id, message=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS: LTX-2 (Image-to-Video)
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/generate/ltx2/i2v", response_model=GenerationResponse)
async def generate_ltx2_i2v(request: LTX2ImageToVideoRequest, http_request: Request):
    """
    Generate a video from an image (LTX-2 Distilled Image-to-Video)
    
    Features:
    - 8-step distilled inference (fast)
    - Native audio generation
    - 24 FPS output
    
    Resolution: Use preset (480p, 720p) or explicit width/height (multiples of 64).
    Duration: Automatically converted to valid frame count (17 + 8*n).
    """
    if model_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if current_model_family != "ltx2":
        raise HTTPException(
            status_code=400,
            detail=f"Wrong model loaded. Need ltx2, got {current_model_family}. Use /reload endpoint."
        )
    
    job_id = str(uuid.uuid4())[:8]
    
    try:
        width, height = resolve_resolution(
            "ltx2",
            resolution_preset=request.resolution_preset,
            width=request.width,
            height=request.height,
        )
        
        # Fetch and resize image
        print(f"📥 Fetching image from: {request.image_url[:80]}...")
        try:
            image_start = await fetch_image_from_url(request.image_url)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to fetch image: {str(e)}")
        
        image_start = image_start.resize((width, height), Image.LANCZOS)
        
        num_frames = duration_to_frames_ltx2(request.duration)
        actual_duration = frames_to_duration(num_frames, LTX2_FPS)
        
        # Validate request won't OOM
        is_valid, warning = validate_generation_request(width, height, num_frames, "ltx2")
        if not is_valid:
            print(f"⚠️ {warning}")
            # Don't fail, but warn - VAE tiling may help
        
        print(f"   Using LTX-2 Dev settings:")
        print(f"   - num_inference_steps: {request.num_inference_steps}")
        print(f"   - input_video_strength: {request.input_video_strength}")
        print(f"   - guidance_scale: {request.guidance_scale}")
        print(f"   - attention_mode: {DETECTED_ATTENTION_MODE}")
        
        output_path, gen_time, metadata = generate_video_internal(
            prompt=request.prompt,
            image_start=image_start,
            width=width,
            height=height,
            num_frames=num_frames,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            input_video_strength=request.input_video_strength,
            seed=request.seed,
            fps=LTX2_FPS,
        )
        
        filename = Path(output_path).name
        job_id = filename.replace(".mp4", "")
        
        # Upload to GCS and get signed URL
        gcs_success, gcs_url, gcs_error = upload_to_gcs(output_path, filename)
        
        if gcs_success:
            # Use the GCS signed URL
            full_url = gcs_url
            # Clean up local file after successful upload
            cleanup_local_file(output_path)
        else:
            # Fallback to local download URL (keep file for local serving)
            print(f"⚠️ GCS upload failed, using local URL: {gcs_error}")
            base_url = str(http_request.base_url).rstrip("/")
            full_url = f"{base_url}/download/{filename}"
        
        return GenerationResponse(
            status="success",
            job_id=job_id,
            output_url=full_url,
            video_url=full_url,
            generation_time_seconds=round(gen_time, 2),
            duration_seconds=actual_duration,
            num_frames=num_frames,
            width=metadata["width"],
            height=metadata["height"],
            seed=metadata["seed"],
        )
        
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        return GenerationResponse(status="error", job_id=job_id, message=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS: WAN2.2 (Image-to-Video Lightning v2)
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/generate/wan22/i2v", response_model=GenerationResponse)
async def generate_wan22_i2v(request: Wan22ImageToVideoRequest, http_request: Request):
    """
    Generate a video from an image (Wan2.2 I2V Lightning v2)
    
    Features:
    - 4-step Lightning v2 inference (ultra-fast)
    - Enhanced prompt comprehension with temporal markers
    - Camera angle and cinematic movement support
    - 16 FPS output
    
    Prompt format for temporal control:
    "(at 0 seconds: description). (at 2 seconds: description)."
    
    Resolution: 480p or 720p (and portrait variants).
    Duration: Automatically converted to valid frame count (5 + 4*n).
    """
    if model_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if current_model_family != "wan22":
        raise HTTPException(
            status_code=400,
            detail=f"Wrong model loaded. Need wan22, got {current_model_family}. Use /reload endpoint."
        )
    
    job_id = str(uuid.uuid4())[:8]
    
    try:
        width, height = resolve_resolution(
            "wan22",
            resolution_preset=request.resolution_preset,
            width=request.width,
            height=request.height,
        )
        
        # Fetch and resize image with aspect ratio preservation
        print(f"📥 Fetching image from: {request.image_url[:80]}...")
        try:
            image_start = await fetch_image_from_url(request.image_url)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to fetch image: {str(e)}")
        
        # Use smart resize with "cover" mode to preserve aspect ratio (like GUI)
        original_size = image_start.size
        image_start = resize_image_preserve_aspect(image_start, width, height, block_size=16, fit_mode="cover")
        print(f"   Resized image: {original_size} -> {image_start.size} (cover mode)")
        
        num_frames = duration_to_frames_wan22(request.duration)
        base_fps = WAN22_FPS
        
        # Determine if we should use sliding window
        # Auto-enable for videos > 5 seconds (81 frames), or if explicitly requested
        use_sliding_window = request.use_sliding_window
        if use_sliding_window is None:
            use_sliding_window = request.duration > 5.0 or request.temporal_upsampling is not None
        
        if use_sliding_window:
            # Use sliding window generation for long videos
            print(f"   Using Enhanced Lightning v2 with SLIDING WINDOW:")
            print(f"   - guidance_phases: {request.guidance_phases}, model_switch_phase: {request.model_switch_phase}")
            print(f"   - guidance_scale: {request.guidance_scale}, guidance2_scale: {request.guidance2_scale}")
            print(f"   - switch_threshold: {request.switch_threshold}")
            print(f"   - flow_shift: {request.flow_shift}, sample_solver: unipc")
            print(f"   - sliding_window_size: {request.sliding_window_size}, overlap: {request.sliding_window_overlap}, overlap_noise: {request.sliding_window_overlap_noise}")
            print(f"   - color_correction: {request.color_correction_strength}, temporal_upsampling: {request.temporal_upsampling}")
            
            # Calculate output fps based on temporal upsampling
            if request.temporal_upsampling == "rife2":
                output_fps = base_fps * 2
            elif request.temporal_upsampling == "rife4":
                output_fps = base_fps * 4
            else:
                output_fps = base_fps
            
            output_path, gen_time, metadata = generate_video_svi2pro_internal(
                prompt=request.prompt,
                image_start=image_start,
                width=width,
                height=height,
                num_frames=num_frames,
                num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance_scale,
                guidance2_scale=request.guidance2_scale,
                guidance_phases=request.guidance_phases,
                model_switch_phase=request.model_switch_phase,
                switch_threshold=request.switch_threshold,
                flow_shift=request.flow_shift,
                seed=request.seed,
                fps=base_fps,
                sliding_window_size=request.sliding_window_size,
                sliding_window_overlap=request.sliding_window_overlap,
                sliding_window_overlap_noise=request.sliding_window_overlap_noise,
                color_correction_strength=request.color_correction_strength,
                temporal_upsampling=request.temporal_upsampling or "",
            )
            actual_duration = frames_to_duration(num_frames, base_fps)
            
            # Account for RIFE upsampling in output frame count
            output_num_frames = num_frames
            if request.temporal_upsampling == "rife2":
                output_num_frames = num_frames * 2 - 1
            elif request.temporal_upsampling == "rife4":
                output_num_frames = num_frames * 4 - 3
        else:
            # Use single-pass generation for short videos
            actual_duration = frames_to_duration(num_frames, WAN22_FPS)
            
            print(f"   Using Enhanced Lightning v2 settings:")
            print(f"   - guidance_phases: {request.guidance_phases}, model_switch_phase: {request.model_switch_phase}")
            print(f"   - guidance_scale: {request.guidance_scale}, guidance2_scale: {request.guidance2_scale}")
            print(f"   - switch_threshold: {request.switch_threshold}")
            print(f"   - flow_shift: {request.flow_shift}, sample_solver: unipc")
            
            output_path, gen_time, metadata = generate_video_internal(
                prompt=request.prompt,
                image_start=image_start,
                width=width,
                height=height,
                num_frames=num_frames,
                num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance_scale,
                guidance2_scale=request.guidance2_scale,
                guidance_phases=request.guidance_phases,
                model_switch_phase=request.model_switch_phase,
                switch_threshold=request.switch_threshold,
                flow_shift=request.flow_shift,
                seed=request.seed,
                fps=WAN22_FPS,
            )
            output_num_frames = num_frames
        
        filename = Path(output_path).name
        job_id = filename.replace(".mp4", "")
        
        # Upload to GCS and get signed URL
        gcs_success, gcs_url, gcs_error = upload_to_gcs(output_path, filename)
        
        if gcs_success:
            # Use the GCS signed URL
            full_url = gcs_url
            # Clean up local file after successful upload
            cleanup_local_file(output_path)
        else:
            # Fallback to local download URL (keep file for local serving)
            print(f"⚠️ GCS upload failed, using local URL: {gcs_error}")
            base_url = str(http_request.base_url).rstrip("/")
            full_url = f"{base_url}/download/{filename}"
        
        return GenerationResponse(
            status="success",
            job_id=job_id,
            output_url=full_url,
            video_url=full_url,
            generation_time_seconds=round(gen_time, 2),
            duration_seconds=actual_duration,
            num_frames=output_num_frames,
            width=metadata["width"],
            height=metadata["height"],
            seed=metadata["seed"],
        )
        
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        return GenerationResponse(status="error", job_id=job_id, message=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS: WAN2.2 SVI2Pro (Image-to-Video with Sliding Window)
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/generate/wan22/i2v", response_model=GenerationResponse)
async def generate_wan22_i2v(request: SVI2ProImageToVideoRequest, http_request: Request):
    """
    Generate video from an image (WAN 2.2)
    
    Features:
    - Sliding window generation for videos longer than 10 seconds
    - 8-step inference
    - RIFE x2 temporal upsampling for smoother output (doubles fps)
    - Color correction between windows for consistency
    - 16 FPS base output (32 FPS with RIFE x2)
    
    Prompt format for temporal control:
    "(at 0 seconds: description). (at 5 seconds: description)."
    
    Resolution: 480p or 720p (and portrait variants).
    """
    if model_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if current_model_family not in ["wan22_svi2pro", "wan22"]:
        raise HTTPException(
            status_code=400,
            detail=f"Wrong model loaded. Need wan22_svi2pro or wan22, got {current_model_family}. Use /reload endpoint."
        )
    
    # Warn if using non-SVI model with SVI2Pro endpoint
    if current_model_family == "wan22":
        print("⚠️ Warning: Using non-SVI2Pro model with SVI2Pro endpoint. For best results, load an SVI2Pro model.")
    
    job_id = str(uuid.uuid4())[:8]
    
    try:
        width, height = resolve_resolution(
            "wan22_svi2pro" if current_model_family == "wan22_svi2pro" else "wan22",
            resolution_preset=request.resolution_preset,
            width=request.width,
            height=request.height,
        )
        
        # Fetch and resize image with aspect ratio preservation (like GUI)
        print(f"📥 Fetching image from: {request.image_url[:80]}...")
        try:
            image_start = await fetch_image_from_url(request.image_url)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to fetch image: {str(e)}")
        
        # Use smart resize with "cover" mode (scale + crop) to preserve aspect ratio
        # This matches the GUI's fit_crop behavior and avoids distortion artifacts
        original_size = image_start.size
        image_start = resize_image_preserve_aspect(image_start, width, height, block_size=16, fit_mode="cover")
        print(f"   Resized image: {original_size} -> {image_start.size} (cover mode)")
        
        num_frames = duration_to_frames_wan22(request.duration)
        base_fps = WAN22_FPS
        
        # Calculate output fps based on temporal upsampling
        if request.temporal_upsampling == "rife2":
            output_fps = base_fps * 2
        elif request.temporal_upsampling == "rife4":
            output_fps = base_fps * 4
        else:
            output_fps = base_fps
        
        actual_duration = frames_to_duration(num_frames, base_fps)
        
        print(f"   Using SVI2Pro Enhanced Lightning v2 settings:")
        print(f"   - guidance_phases: {request.guidance_phases}, model_switch_phase: {request.model_switch_phase}")
        print(f"   - guidance_scale: {request.guidance_scale}, guidance2_scale: {request.guidance2_scale}")
        print(f"   - switch_threshold: {request.switch_threshold}")
        print(f"   - flow_shift: {request.flow_shift}, sample_solver: unipc")
        print(f"   - sliding_window_size: {request.sliding_window_size}, overlap: {request.sliding_window_overlap}, overlap_noise: {request.sliding_window_overlap_noise}")
        print(f"   - color_correction_strength: {request.color_correction_strength}")
        print(f"   - temporal_upsampling: {request.temporal_upsampling}")
        print(f"   - base_fps: {base_fps}, output_fps: {output_fps}")
        
        # Use the dedicated SVI2Pro internal function which properly handles all SVI2Pro parameters
        # including overlap_size, color_correction_strength, and proper prefix_video setup
        output_path, gen_time, metadata = generate_video_svi2pro_internal(
            prompt=request.prompt,
            image_start=image_start,
            width=width,
            height=height,
            num_frames=num_frames,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            guidance2_scale=request.guidance2_scale,
            guidance_phases=request.guidance_phases,
            model_switch_phase=request.model_switch_phase,
            switch_threshold=request.switch_threshold,
            flow_shift=request.flow_shift,
            seed=request.seed,
            fps=base_fps,
            sliding_window_size=request.sliding_window_size,
            sliding_window_overlap=request.sliding_window_overlap,
            sliding_window_overlap_noise=request.sliding_window_overlap_noise,
            color_correction_strength=request.color_correction_strength,
            temporal_upsampling=request.temporal_upsampling or "",
        )
        
        filename = Path(output_path).name
        job_id = filename.replace(".mp4", "")
        
        # Upload to GCS and get signed URL
        gcs_success, gcs_url, gcs_error = upload_to_gcs(output_path, filename)
        
        if gcs_success:
            full_url = gcs_url
            # Clean up local file after successful upload
            cleanup_local_file(output_path)
        else:
            # Fallback to local download URL (keep file for local serving)
            print(f"⚠️ GCS upload failed, using local URL: {gcs_error}")
            base_url = str(http_request.base_url).rstrip("/")
            full_url = f"{base_url}/download/{filename}"
        
        # Account for RIFE upsampling in output frame count
        output_num_frames = num_frames
        if request.temporal_upsampling == "rife2":
            output_num_frames = num_frames * 2 - 1
        elif request.temporal_upsampling == "rife4":
            output_num_frames = num_frames * 4 - 3
        
        return GenerationResponse(
            status="success",
            job_id=job_id,
            output_url=full_url,
            video_url=full_url,
            generation_time_seconds=round(gen_time, 2),
            duration_seconds=actual_duration,
            num_frames=output_num_frames,
            width=metadata["width"],
            height=metadata["height"],
            seed=metadata["seed"],
        )
        
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        return GenerationResponse(status="error", job_id=job_id, message=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS: GENERIC (Legacy/Base64)
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/generate/i2v", response_model=GenerationResponse)
async def generate_i2v_legacy(request: LTX2ImageToVideoRequest, http_request: Request):
    """Legacy endpoint - routes to current model's I2V"""
    if current_model_family == "ltx2":
        return await generate_ltx2_i2v(request, http_request)
    elif current_model_family in ["wan22", "wan22_svi2pro"]:
        # Convert request
        wan_request = Wan22ImageToVideoRequest(
            prompt=request.prompt,
            image_url=request.image_url,
            duration=request.duration,
            resolution_preset=request.resolution_preset if request.resolution_preset in ["480p", "720p"] else "480p",
            guidance_scale=1.0,
            seed=request.seed,
        )
        return await generate_wan22_i2v(wan_request, http_request)
    else:
        raise HTTPException(status_code=400, detail=f"Model family {current_model_family} doesn't support I2V")


@app.post("/generate/i2v-base64", response_model=GenerationResponse)
async def generate_i2v_base64(request: ImageToVideoRequest, http_request: Request):
    """Generate video from base64 image"""
    if model_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    job_id = str(uuid.uuid4())[:8]
    
    try:
        image_start = decode_base64_image(request.image_base64)
        image_start = image_start.resize((request.width, request.height), Image.LANCZOS)
        
        if current_model_family == "ltx2":
            num_frames = duration_to_frames_ltx2(request.duration)
            fps = LTX2_FPS
        else:
            num_frames = duration_to_frames_wan22(request.duration)
            fps = WAN22_FPS
        
        output_path, gen_time, metadata = generate_video_internal(
            prompt=request.prompt,
            image_start=image_start,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            num_frames=num_frames,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            seed=request.seed,
            fps=fps,
        )
        
        filename = Path(output_path).name
        actual_duration = frames_to_duration(num_frames, fps)
        base_url = str(http_request.base_url).rstrip("/")
        full_url = f"{base_url}/download/{filename}"
        
        return GenerationResponse(
            status="success",
            job_id=filename.replace(".mp4", ""),
            output_url=full_url,
            video_url=full_url,
            generation_time_seconds=round(gen_time, 2),
            duration_seconds=actual_duration,
            num_frames=num_frames,
        )
        
    except Exception as e:
        traceback.print_exc()
        return GenerationResponse(status="error", job_id=job_id, message=str(e))


@app.post("/generate/t2v", response_model=GenerationResponse)
async def generate_t2v(request: TextToVideoRequest, http_request: Request):
    """Generate video from text prompt"""
    if model_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    job_id = str(uuid.uuid4())[:8]
    
    try:
        if current_model_family == "ltx2":
            num_frames = duration_to_frames_ltx2(request.duration)
            fps = LTX2_FPS
        else:
            num_frames = duration_to_frames_wan22(request.duration)
            fps = WAN22_FPS
        
        output_path, gen_time, metadata = generate_video_internal(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            num_frames=num_frames,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            seed=request.seed,
            fps=fps,
        )
        
        filename = Path(output_path).name
        actual_duration = frames_to_duration(num_frames, fps)
        base_url = str(http_request.base_url).rstrip("/")
        full_url = f"{base_url}/download/{filename}"
        
        return GenerationResponse(
            status="success",
            job_id=filename.replace(".mp4", ""),
            output_url=full_url,
            video_url=full_url,
            generation_time_seconds=round(gen_time, 2),
            duration_seconds=actual_duration,
            num_frames=num_frames,
        )
        
    except Exception as e:
        traceback.print_exc()
        return GenerationResponse(status="error", job_id=job_id, message=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS: MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/download/{filename}")
async def download_file(filename: str):
    """Download a generated file by filename"""
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    media_type = "video/mp4" if filename.endswith(".mp4") else "image/png"
    return FileResponse(str(file_path), media_type=media_type, filename=filename)


def cleanup_local_file(file_path: str, delay_seconds: float = 0) -> bool:
    """
    Delete a local file after successful GCS upload.
    
    Args:
        file_path: Path to the file to delete
        delay_seconds: Optional delay before deletion (for async cleanup)
        
    Returns:
        bool: True if deleted successfully
    """
    try:
        path = Path(file_path)
        if path.exists():
            if delay_seconds > 0:
                time.sleep(delay_seconds)
            path.unlink()
            print(f"🗑️ Cleaned up local file: {path.name}")
            return True
    except Exception as e:
        print(f"⚠️ Failed to cleanup {file_path}: {e}")
    return False


@app.delete("/files/{job_id}")
async def delete_file(job_id: str):
    """Delete a generated file"""
    deleted = False
    for ext in [".mp4", ".png"]:
        file_path = OUTPUT_DIR / f"{job_id}{ext}"
        if file_path.exists():
            file_path.unlink()
            deleted = True
            print(f"🗑️ Deleted file: {file_path.name}")
    
    if deleted:
        return {"status": "deleted", "job_id": job_id}
    raise HTTPException(status_code=404, detail="File not found")


@app.post("/reload")
async def reload_model(
    model_type: str = DEFAULT_MODEL_TYPE, 
    profile: int = DEFAULT_PROFILE
):
    """
    Reload the model (for switching between model types)
    
    Available model types:
    - ltx2_distilled: LTX-2 Image-to-Video
    - z_image: Z-Image Text-to-Image
    - i2v_2_2_Enhanced_Lightning_v2: Wan2.2 I2V Enhanced Lightning v2
    """
    global current_model_type, current_model_family
    
    try:
        unload_model()
        load_wan2gp_model(model_type, profile)
        return {
            "status": "success", 
            "model_type": current_model_type,
            "model_family": current_model_family,
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # CRITICAL: Set environment variables BEFORE uvicorn re-imports the module
    # When uvicorn.run() is called with a string import path, it re-imports api_server.py
    # At that point sys.argv is cleared, so we need env vars to pass the config
    os.environ["WAN2GP_MODEL_TYPE"] = API_MODEL_TYPE
    os.environ["WAN2GP_PROFILE"] = str(API_PROFILE)
    
    print(f"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                     WAN2GP MULTI-MODEL API SERVER                             ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║  Host:       {API_HOST:<60} ║
║  Port:       {API_PORT:<60} ║
║  Model:      {API_MODEL_TYPE:<60} ║
║  Profile:    {API_PROFILE:<60} ║
║  Output Dir: {str(OUTPUT_DIR):<60} ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║  Endpoints:                                                                   ║
║    POST /generate/image       - Z-Image text-to-image                         ║
║    POST /generate/ltx2/i2v    - LTX-2 image-to-video                          ║
║    POST /generate/wan22/i2v   - Wan2.2 Lightning v2 image-to-video            ║
║    POST /generate/wan22/i2v - WAN22 I2V (sliding window, RIFE x2)             ║
║    POST /reload               - Switch model type                             ║
║    GET  /health               - Health check                                  ║
║    GET  /info                 - API information                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(
        "api_server:app",
        host=API_HOST,
        port=API_PORT,
        reload=API_RELOAD,
        workers=1,
    )


if __name__ == "__main__":
    main()
