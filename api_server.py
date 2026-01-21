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
        "480p": (848, 480),
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
    
    resolution_preset: Optional[Literal["480p", "480p_portrait", "720p", "720p_portrait", "576p", "576p_portrait"]] = Field(
        "480p",
        description="Resolution preset: 480p, 720p, 576p (and portrait variants)"
    )
    width: Optional[int] = Field(None, description="Video width (must be multiple of 16)")
    height: Optional[int] = Field(None, description="Video height (must be multiple of 16)")
    
    # Lightning v2 uses fixed 4 steps and guidance 1.0
    num_inference_steps: int = Field(4, ge=4, le=8, description="Inference steps (4 for Lightning v2)")
    guidance_scale: float = Field(1.0, ge=1.0, le=5.0, description="CFG scale (1.0 for Lightning v2)")
    flow_shift: float = Field(5.0, ge=1.0, le=15.0, description="Flow shift parameter")
    seed: int = Field(-1, description="Random seed (-1 for random)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "(at 0 seconds: wide shot of a woman standing, cinematic lighting). (at 2 seconds: camera slowly zooms in). (at 4 seconds: close-up on face, she smiles).",
                "image_url": "https://example.com/portrait.jpg",
                "duration": 5.0,
                "resolution_preset": "480p",
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


def generate_video_internal(
    prompt: str,
    image_start: Optional[Image.Image] = None,
    negative_prompt: str = "",
    width: int = 768,
    height: int = 512,
    num_frames: int = 121,
    num_inference_steps: int = 8,
    guidance_scale: float = 4.0,
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
    
    # Build generation kwargs
    gen_kwargs = {
        "input_prompt": prompt,
        "n_prompt": negative_prompt if negative_prompt else None,
        "image_start": image_start,
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
    }
    
    # Add flow_shift for Wan2.2
    if flow_shift is not None:
        gen_kwargs["flow_shift"] = flow_shift
    
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
        
        output_path, gen_time, metadata = generate_video_internal(
            prompt=request.prompt,
            image_start=image_start,
            width=width,
            height=height,
            num_frames=num_frames,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            flow_shift=request.flow_shift,
            seed=request.seed,
            fps=WAN22_FPS,
        )
        
        filename = Path(output_path).name
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
