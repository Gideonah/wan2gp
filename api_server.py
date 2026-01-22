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
    parser.add_argument("--model-type", type=str, default=os.environ.get("WAN2GP_MODEL_TYPE", "ltx2_distilled"))
    parser.add_argument("--profile", type=int, default=int(os.environ.get("WAN2GP_PROFILE", "3")))
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
}

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
        "720p": (1280, 720),
        "720p_portrait": (720, 1280),
        "576p": (1024, 576),
        "576p_portrait": (576, 1024),
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
current_model_family = None
offloadobj = None

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
    seed: int = Field(-1, description="Random seed (-1 for random)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "A person slowly turning their head and smiling",
                "image_url": "https://example.com/portrait.jpg",
                "duration": 5.0,
                "resolution_preset": "720p",
                "guidance_scale": 4.0,
                "seed": -1
            }
        }


class Wan22ImageToVideoRequest(BaseModel):
    """Request model for Wan2.2 I2V Lightning v2 generation"""
    prompt: str = Field(..., description="Text prompt describing the video (supports temporal markers)")
    image_url: str = Field(..., description="URL of the input image to animate")
    duration: float = Field(5.0, ge=0.3, le=15.0, description="Video duration in seconds")
    
    resolution_preset: Optional[Literal["default", "480p", "480p_portrait", "720p", "720p_portrait", "576p", "576p_portrait"]] = Field(
        "default",
        description="Resolution preset: default (832x480), 480p, 720p, 576p (and portrait variants)"
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
    
    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "(at 0 seconds: wide shot of a woman standing, cinematic lighting). (at 2 seconds: camera slowly zooms in). (at 4 seconds: close-up on face, she smiles).",
                "image_url": "https://example.com/portrait.jpg",
                "duration": 5.0,
                "resolution_preset": "default",
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
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()
        image_bytes = response.content
        image = Image.open(io.BytesIO(image_bytes))
        return image.convert("RGB")


def decode_base64_image(image_base64: str) -> Image.Image:
    """Decode a base64 string to PIL Image"""
    if "," in image_base64:
        image_base64 = image_base64.split(",")[1]
    image_bytes = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image_bytes))
    return image.convert("RGB")


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_wan2gp_model(model_type: str = DEFAULT_MODEL_TYPE, profile: int = DEFAULT_PROFILE):
    """Load the Wan2GP model into VRAM"""
    global model_instance, model_handler, model_def, current_model_type, current_model_family, offloadobj
    
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
    current_model_family = get_model_family(model_type)
    
    load_time = time.time() - start_time
    print(f"✅ Model loaded in {load_time:.1f}s (family: {current_model_family})")
    
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"   GPU: {props.name}")
        print(f"   VRAM: {props.total_memory // 1024 // 1024}MB")
    
    return model_instance


def unload_model():
    """Release model from memory"""
    global model_instance, offloadobj
    
    if offloadobj is not None:
        offloadobj.release()
        offloadobj = None
    
    model_instance = None
    gc.collect()
    torch.cuda.empty_cache()


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
    
    # Set up offload shared state
    offload.shared_state["_attention"] = "sdpa"
    offload.shared_state["_chipmunk"] = False
    offload.shared_state["_radial"] = False
    offload.shared_state["_nag_scale"] = 1.0
    offload.shared_state["_nag_tau"] = 3.5
    offload.shared_state["_nag_alpha"] = 0.5
    
    model_instance._interrupt = False
    
    loras_slists = {
        "phase1": [], "phase2": [], "phase3": [], "shared": [],
        "model_switch_step": num_inference_steps,
        "model_switch_step2": num_inference_steps,
    }
    
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
            loras_slists=loras_slists,
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
    if image_start is not None:
        print(f"   Input image: {image_start.size[0]}x{image_start.size[1]}")
    
    start_time = time.time()
    
    # Set up offload shared state
    offload.shared_state["_attention"] = "sdpa"
    offload.shared_state["_chipmunk"] = False
    offload.shared_state["_radial"] = False
    offload.shared_state["_nag_scale"] = 1.0
    offload.shared_state["_nag_tau"] = 3.5
    offload.shared_state["_nag_alpha"] = 0.5
    
    model_instance._interrupt = False
    
    loras_slists = {
        "phase1": [], "phase2": [], "phase3": [], "shared": [],
        "model_switch_step": num_inference_steps,
        "model_switch_step2": num_inference_steps,
    }
    
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
    if image_start is not None and current_model_family == "wan22":
        # For Wan2.2 i2v: convert PIL image to tensor and pass as input_video
        image_start_tensor = convert_pil_image_to_tensor(image_start)
        # Add time dimension: (C, H, W) -> (C, 1, H, W)
        input_video_tensor = image_start_tensor.unsqueeze(1)
        print(f"   Converted to input_video tensor: {input_video_tensor.shape}")
    
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
        "VAE_tile_size": 0,
        "loras_slists": loras_slists,
        "callback": video_progress_callback,
    }
    
    # For Wan2.2 i2v models, pass the image as input_video (required for i2v_class flow)
    if input_video_tensor is not None:
        gen_kwargs["input_video"] = input_video_tensor
    elif image_start is not None and current_model_family != "wan22":
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
    
    # For Lightning models, use euler solver (default is unipc which is slower and lower quality for distilled models)
    if current_model_family == "wan22":
        gen_kwargs["sample_solver"] = "euler"
    
    try:
        # Initialize cache attribute (required by any2video.py for step-skipping logic)
        # Set to None to disable step-skipping cache (TeaCache/MagCache)
        if hasattr(model_instance, 'model') and model_instance.model is not None:
            model_instance.model.cache = None
        
        result = model_instance.generate(**gen_kwargs)
        
        # Extract video tensor and audio
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
        
        # Mux audio if present
        if audio_data is not None:
            from postprocessing.mmaudio.data.av_utils import remux_with_audio
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
            temp_video_path = output_path.with_name(output_path.stem + '_tmp.mp4')
            output_path.rename(temp_video_path)
            remux_with_audio(temp_video_path, output_path, audio_tensor, audio_sr)
            temp_video_path.unlink(missing_ok=True)
        
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"Generation failed: {str(e)}")
    finally:
        gc.collect()
        torch.cuda.empty_cache()
    
    generation_time = time.time() - start_time
    print(f"✅ Video saved to {output_path} in {generation_time:.1f}s")
    
    metadata = {
        "num_frames": num_frames,
        "duration": frames_to_duration(num_frames, fps),
        "fps": fps,
        "width": width,
        "height": height,
        "seed": seed,
    }
    
    return str(output_path), generation_time, metadata


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
        
        output_path, gen_time, metadata = generate_video_internal(
            prompt=request.prompt,
            image_start=image_start,
            width=width,
            height=height,
            num_frames=num_frames,
            num_inference_steps=8,  # LTX-2 distilled uses 8 steps
            guidance_scale=request.guidance_scale,
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
        else:
            # Fallback to local download URL
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
        
        # Fetch and resize image
        print(f"📥 Fetching image from: {request.image_url[:80]}...")
        try:
            image_start = await fetch_image_from_url(request.image_url)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to fetch image: {str(e)}")
        
        image_start = image_start.resize((width, height), Image.LANCZOS)
        
        num_frames = duration_to_frames_wan22(request.duration)
        actual_duration = frames_to_duration(num_frames, WAN22_FPS)
        
        print(f"   Using Enhanced Lightning v2 settings:")
        print(f"   - guidance_phases: {request.guidance_phases}, model_switch_phase: {request.model_switch_phase}")
        print(f"   - guidance_scale: {request.guidance_scale}, guidance2_scale: {request.guidance2_scale}")
        print(f"   - switch_threshold: {request.switch_threshold}")
        print(f"   - flow_shift: {request.flow_shift}, sample_solver: euler")
        
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
        
        filename = Path(output_path).name
        job_id = filename.replace(".mp4", "")
        
        # Upload to GCS and get signed URL
        gcs_success, gcs_url, gcs_error = upload_to_gcs(output_path, filename)
        
        if gcs_success:
            # Use the GCS signed URL
            full_url = gcs_url
        else:
            # Fallback to local download URL
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
# ENDPOINTS: GENERIC (Legacy/Base64)
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/generate/i2v", response_model=GenerationResponse)
async def generate_i2v_legacy(request: LTX2ImageToVideoRequest, http_request: Request):
    """Legacy endpoint - routes to current model's I2V"""
    if current_model_family == "ltx2":
        return await generate_ltx2_i2v(request, http_request)
    elif current_model_family == "wan22":
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


@app.delete("/files/{job_id}")
async def delete_file(job_id: str):
    """Delete a generated file"""
    for ext in [".mp4", ".png"]:
        file_path = OUTPUT_DIR / f"{job_id}{ext}"
    if file_path.exists():
        file_path.unlink()
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
║    POST /generate/image      - Z-Image text-to-image                          ║
║    POST /generate/ltx2/i2v   - LTX-2 image-to-video                           ║
║    POST /generate/wan22/i2v  - Wan2.2 Lightning v2 image-to-video             ║
║    POST /reload              - Switch model type                              ║
║    GET  /health              - Health check                                   ║
║    GET  /info                - API information                                ║
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
