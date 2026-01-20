#!/usr/bin/env python3
"""
Wan2GP API Server for Serverless Deployment (Vast.ai, RunPod, Modal)

This FastAPI server provides a REST API interface to Wan2GP's video generation
capabilities, designed for serverless GPU deployments.

Optimized for LTX-2 Distilled Image-to-Video generation.

Usage:
    python api_server.py [--port 8000] [--model-type ltx2_distilled] [--profile 5]

Environment Variables:
    WAN2GP_MODEL_TYPE: Default model type to load (default: ltx2_distilled)
    WAN2GP_PROFILE: MMGP profile for memory optimization (1-6, default: 5)
    WAN2GP_OUTPUT_DIR: Directory to save generated videos (default: /workspace/outputs)
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
from typing import Optional, List, Union
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
    parser.add_argument("--profile", type=int, default=int(os.environ.get("WAN2GP_PROFILE", "5")))
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("-h", "--help", action="store_true")
    
    args, unknown = parser.parse_known_args()
    
    if args.help:
        print("""
Wan2GP API Server (LTX-2 Distilled)

Usage:
    python api_server.py [OPTIONS]

Options:
    --host          Host to bind to (default: 0.0.0.0)
    --port          Port to listen on (default: 8000)
    --model-type    Model type to load (default: ltx2_distilled)
    --profile       MMGP memory profile 1-6 (default: 5)
    --reload        Enable auto-reload for development

Environment Variables:
    WAN2GP_MODEL_TYPE   Default model type (default: ltx2_distilled)
    WAN2GP_PROFILE      Default profile
    WAN2GP_OUTPUT_DIR   Output directory for videos
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
# Keep only the script name
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
from fastapi import FastAPI, HTTPException, BackgroundTasks
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

# LTX-2 specific settings
LTX2_FPS = 24  # LTX-2 native FPS
LTX2_MIN_FRAMES = 17  # Minimum frames
LTX2_FRAME_STEP = 8  # Frames increment in steps of 8
LTX2_RESOLUTION_DIVISOR = 64  # Must be divisible by 64 for distilled pipeline

# Resolution presets for LTX-2
RESOLUTION_PRESETS = {
    "480p": (832, 480),       # ~16:9 landscape
    "480p_portrait": (480, 832),
    "720p": (1280, 720),      # 16:9 HD
    "720p_portrait": (720, 1280),
    "768": (768, 768),        # Square
    "1024": (1024, 1024),     # Square HD
    "landscape": (1024, 576), # 16:9
    "portrait": (576, 1024),  # 9:16
    "wide": (1280, 576),      # Ultra-wide
}

# Global model reference
model_instance = None
model_handler = None
model_def = None
current_model_type = None
offloadobj = None

# ═══════════════════════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class LTX2ImageToVideoRequest(BaseModel):
    """Request model for LTX-2 image-to-video generation with URL input"""
    prompt: str = Field(..., description="Text prompt describing the video motion/action")
    image_url: str = Field(..., description="URL of the input image to animate")
    duration: float = Field(5.0, ge=0.7, le=20.0, description="Video duration in seconds (0.7-20)")
    
    # Resolution options - use EITHER preset OR width/height
    resolution_preset: Optional[str] = Field(
        None, 
        description="Resolution preset: 480p, 720p, 768, 1024, landscape, portrait, wide"
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


class TextToVideoRequest(BaseModel):
    """Request model for text-to-video generation"""
    prompt: str = Field(..., description="Text prompt describing the video")
    negative_prompt: str = Field("", description="Negative prompt")
    duration: float = Field(5.0, ge=0.7, le=20.0, description="Video duration in seconds")
    width: int = Field(768, description="Video width (must be multiple of 64)")
    height: int = Field(512, description="Video height (must be multiple of 64)")
    num_inference_steps: int = Field(8, description="Number of denoising steps (8 for distilled)")
    guidance_scale: float = Field(4.0, description="Classifier-free guidance scale")
    seed: int = Field(-1, description="Random seed (-1 for random)")


class ImageToVideoRequest(BaseModel):
    """Request model for image-to-video generation with base64"""
    prompt: str = Field(..., description="Text prompt describing the video")
    image_base64: str = Field(..., description="Base64 encoded start image (PNG/JPEG)")
    negative_prompt: str = Field("", description="Negative prompt")
    duration: float = Field(5.0, ge=0.7, le=20.0, description="Video duration in seconds")
    width: int = Field(768, description="Video width (must be multiple of 64)")
    height: int = Field(512, description="Video height (must be multiple of 64)")
    num_inference_steps: int = Field(8, description="Number of denoising steps (8 for distilled)")
    guidance_scale: float = Field(4.0, description="Classifier-free guidance scale")
    seed: int = Field(-1, description="Random seed (-1 for random)")


class GenerationResponse(BaseModel):
    """Response model for video generation"""
    status: str
    job_id: str
    message: Optional[str] = None
    video_url: Optional[str] = None
    generation_time_seconds: Optional[float] = None
    duration_seconds: Optional[float] = None
    num_frames: Optional[int] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_type: Optional[str] = None
    gpu_name: Optional[str] = None
    gpu_memory_total_mb: Optional[int] = None
    gpu_memory_used_mb: Optional[int] = None


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def duration_to_frames(duration_seconds: float, fps: int = LTX2_FPS) -> int:
    """
    Convert duration in seconds to valid frame count for LTX-2.
    LTX-2 requires frames = 17 + 8*n (minimum 17, steps of 8)
    """
    target_frames = int(duration_seconds * fps)
    
    # LTX-2: frames must be 17 + 8*n
    if target_frames < LTX2_MIN_FRAMES:
        return LTX2_MIN_FRAMES
    
    # Find nearest valid frame count: 17, 25, 33, 41, 49, ...
    n = max(0, (target_frames - LTX2_MIN_FRAMES) // LTX2_FRAME_STEP)
    valid_frames = LTX2_MIN_FRAMES + (n * LTX2_FRAME_STEP)
    
    # Check if rounding up is closer
    next_valid = valid_frames + LTX2_FRAME_STEP
    if abs(next_valid - target_frames) < abs(valid_frames - target_frames):
        valid_frames = next_valid
    
    return valid_frames


def resolve_resolution(
    resolution_preset: Optional[str] = None,
    width: Optional[int] = None, 
    height: Optional[int] = None,
    default_width: int = 768,
    default_height: int = 512,
) -> tuple[int, int]:
    """
    Resolve resolution from preset or explicit width/height.
    Returns (width, height) aligned to LTX2_RESOLUTION_DIVISOR (64).
    """
    # Use preset if provided
    if resolution_preset and resolution_preset in RESOLUTION_PRESETS:
        return RESOLUTION_PRESETS[resolution_preset]
    
    # Use explicit values or defaults
    w = width if width is not None else default_width
    h = height if height is not None else default_height
    
    # Align to divisor (64 for distilled)
    w = (w // LTX2_RESOLUTION_DIVISOR) * LTX2_RESOLUTION_DIVISOR
    h = (h // LTX2_RESOLUTION_DIVISOR) * LTX2_RESOLUTION_DIVISOR
    
    # Ensure minimum size
    w = max(256, w)
    h = max(256, h)
    
    return w, h


def frames_to_duration(num_frames: int, fps: int = LTX2_FPS) -> float:
    """Convert frame count to duration in seconds"""
    return round(num_frames / fps, 2)


async def fetch_image_from_url(url: str, timeout: float = 30.0) -> Image.Image:
    """Fetch an image from a URL and return as PIL Image"""
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()
        
        content_type = response.headers.get("content-type", "")
        if not content_type.startswith("image/"):
            # Try to open anyway, PIL will validate
            pass
        
        image_bytes = response.content
        image = Image.open(io.BytesIO(image_bytes))
        return image.convert("RGB")


def decode_base64_image(image_base64: str) -> Image.Image:
    """Decode a base64 string to PIL Image"""
    # Handle data URL format
    if "," in image_base64:
        image_base64 = image_base64.split(",")[1]
    
    image_bytes = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image_bytes))
    return image.convert("RGB")


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_wan2gp_model(model_type: str = DEFAULT_MODEL_TYPE, profile: int = DEFAULT_PROFILE):
    """
    Load the Wan2GP model into VRAM.
    This is called once at startup to keep the model warm.
    """
    global model_instance, model_handler, model_def, current_model_type, offloadobj
    
    print(f"⏳ Loading model: {model_type} (profile: {profile})...")
    start_time = time.time()
    
    # Import wgp functions after setting up paths
    from wgp import (
        load_models, 
        get_model_def, 
        get_base_model_type, 
        get_model_handler,
    )
    
    # Get model definitions
    model_def = get_model_def(model_type)
    base_model_type = get_base_model_type(model_type)
    model_handler = get_model_handler(base_model_type)
    
    # Load the model with the specified profile
    # override_profile parameter controls memory optimization level
    model_instance, offloadobj = load_models(model_type, override_profile=profile)
    current_model_type = model_type
    
    load_time = time.time() - start_time
    print(f"✅ Model loaded in {load_time:.1f}s")
    
    # Print GPU info
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
# VIDEO GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_video_internal(
    prompt: str,
    image_start: Optional[Image.Image] = None,
    negative_prompt: str = "",
    width: int = 768,
    height: int = 512,
    num_frames: int = 121,
    num_inference_steps: int = 8,
    guidance_scale: float = 4.0,
    seed: int = -1,
    fps: int = LTX2_FPS,
) -> tuple[str, float, dict]:
    """
    Internal function to generate video using the loaded model.
    Returns (output_path, generation_time, metadata)
    """
    global model_instance, model_def, current_model_type
    
    if model_instance is None:
        raise RuntimeError("Model not loaded")
    
    # Validate dimensions for LTX-2 (must be multiples of 64)
    width = (width // 64) * 64
    height = (height // 64) * 64
    
    # Ensure minimum size
    width = max(256, width)
    height = max(256, height)
    
    # Handle seed
    if seed < 0:
        seed = int(torch.randint(0, 2**32 - 1, (1,)).item())
    
    # Create job ID and output path
    job_id = str(uuid.uuid4())[:8]
    output_path = OUTPUT_DIR / f"{job_id}.mp4"
    
    print(f"🎬 Generating video: {prompt[:50]}...")
    print(f"   Resolution: {width}x{height}, Frames: {num_frames}, Steps: {num_inference_steps}")
    print(f"   Duration: {frames_to_duration(num_frames, fps)}s @ {fps}fps")
    
    start_time = time.time()
    
    # Set up offload shared state (required by generate)
    offload.shared_state["_attention"] = "sdpa"  # Default attention mode
    offload.shared_state["_chipmunk"] = False
    offload.shared_state["_radial"] = False
    offload.shared_state["_nag_scale"] = 1.0
    offload.shared_state["_nag_tau"] = 3.5
    offload.shared_state["_nag_alpha"] = 0.5
    
    # Set interrupt flag
    model_instance._interrupt = False
    
    # Empty loras configuration (no loras active)
    loras_slists = {
        "phase1": [],
        "phase2": [],
        "phase3": [],
        "shared": [],
        "model_switch_step": num_inference_steps,
        "model_switch_step2": num_inference_steps,
    }
    
    # Progress callback
    def progress_callback(*args, **kwargs):
        step = args[0] if len(args) > 0 else -1
        if step >= 0:
            print(f"   Step {step + 1}/{num_inference_steps}")
    
    # Run generation
    try:
        result = model_instance.generate(
            input_prompt=prompt,
            n_prompt=negative_prompt if negative_prompt else None,
            image_start=image_start,
            image_end=None,
            width=width,
            height=height,
            frame_num=num_frames,
            sampling_steps=num_inference_steps,
            guide_scale=guidance_scale,
            seed=seed,
            fps=float(fps),
            callback=progress_callback,
            VAE_tile_size=0,  # Auto-detect
            loras_slists=loras_slists,
        )
        
        # Extract video tensor and audio
        if isinstance(result, dict):
            video_tensor = result.get("x", result)
            audio_data = result.get("audio", None)
            audio_sr = result.get("audio_sampling_rate", 48000)
        else:
            video_tensor = result
            audio_data = None
            audio_sr = 48000
        
        # Save video (with audio if available)
        if audio_data is not None:
            save_video(video_tensor, str(output_path), fps=fps, audio=audio_data, audio_sample_rate=audio_sr)
        else:
            save_video(video_tensor, str(output_path), fps=fps)
        
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"Generation failed: {str(e)}")
    
    finally:
        # Cleanup
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
    # Startup: Load model
    print("🚀 Starting Wan2GP API Server (LTX-2 Distilled)...")
    try:
        load_wan2gp_model()
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        traceback.print_exc()
    
    yield
    
    # Shutdown: Unload model
    print("🛑 Shutting down...")
    unload_model()


app = FastAPI(
    title="LTX-2 Video Generation API",
    description="REST API for LTX-2 Distilled image-to-video generation",
    version="2.0.0",
    lifespan=lifespan,
)

# Mount outputs directory for video downloads
app.mount("/download", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")


@app.get("/", response_model=HealthResponse)
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    gpu_name = None
    gpu_memory_total = None
    gpu_memory_used = None
    
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu_name = props.name
        gpu_memory_total = props.total_memory // 1024 // 1024
        gpu_memory_used = torch.cuda.memory_allocated(0) // 1024 // 1024
    
    return HealthResponse(
        status="healthy" if model_instance is not None else "degraded",
        model_loaded=model_instance is not None,
        model_type=current_model_type,
        gpu_name=gpu_name,
        gpu_memory_total_mb=gpu_memory_total,
        gpu_memory_used_mb=gpu_memory_used,
    )


@app.post("/generate/i2v", response_model=GenerationResponse)
async def generate_image_to_video_url(request: LTX2ImageToVideoRequest):
    """
    Generate a video from an image URL (LTX-2 Image-to-Video)
    
    This endpoint:
    - Fetches the image from the provided URL
    - Generates a video with the specified duration
    - Returns a download URL for the generated video
    
    Resolution options:
    - Use `resolution_preset`: "480p", "720p", "768", "1024", "landscape", "portrait", "wide"
    - OR specify `width` and `height` directly (must be multiples of 64)
    
    Duration is automatically converted to the nearest valid frame count for LTX-2.
    """
    if model_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    job_id = str(uuid.uuid4())[:8]
    
    try:
        # Resolve resolution from preset or explicit values
        width, height = resolve_resolution(
            resolution_preset=request.resolution_preset,
            width=request.width,
            height=request.height,
            default_width=768,
            default_height=512,
        )
        
        # Fetch image from URL
        print(f"📥 Fetching image from: {request.image_url[:80]}...")
        try:
            image_start = await fetch_image_from_url(request.image_url)
            print(f"   Original image size: {image_start.size}")
        except httpx.HTTPError as e:
            raise HTTPException(status_code=400, detail=f"Failed to fetch image: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")
        
        # Resize image to target dimensions
        image_start = image_start.resize((width, height), Image.LANCZOS)
        print(f"   Resized to: {width}x{height}")
        
        # Convert duration to valid frame count
        num_frames = duration_to_frames(request.duration)
        actual_duration = frames_to_duration(num_frames)
        
        # LTX-2 distilled uses 8 inference steps
        num_inference_steps = 8
        
        output_path, gen_time, metadata = generate_video_internal(
            prompt=request.prompt,
            image_start=image_start,
            negative_prompt="",
            width=width,
            height=height,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=request.guidance_scale,
            seed=request.seed,
            fps=LTX2_FPS,
        )
        
        filename = Path(output_path).name
        return GenerationResponse(
            status="success",
            job_id=filename.replace(".mp4", ""),
            video_url=f"/download/{filename}",
            generation_time_seconds=round(gen_time, 2),
            duration_seconds=actual_duration,
            num_frames=num_frames,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        return GenerationResponse(
            status="error",
            job_id=job_id,
            message=str(e),
        )


@app.post("/generate/i2v-base64", response_model=GenerationResponse)
async def generate_image_to_video_base64(request: ImageToVideoRequest):
    """
    Generate a video from a base64-encoded image (alternative endpoint)
    """
    if model_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    job_id = str(uuid.uuid4())[:8]
    
    try:
        # Decode the input image
        image_start = decode_base64_image(request.image_base64)
        
        # Resize to target dimensions
        image_start = image_start.resize((request.width, request.height), Image.LANCZOS)
        
        # Convert duration to valid frame count
        num_frames = duration_to_frames(request.duration)
        actual_duration = frames_to_duration(num_frames)
        
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
            fps=LTX2_FPS,
        )
        
        filename = Path(output_path).name
        return GenerationResponse(
            status="success",
            job_id=filename.replace(".mp4", ""),
            video_url=f"/download/{filename}",
            generation_time_seconds=round(gen_time, 2),
            duration_seconds=actual_duration,
            num_frames=num_frames,
        )
        
    except Exception as e:
        traceback.print_exc()
        return GenerationResponse(
            status="error",
            job_id=job_id,
            message=str(e),
        )


@app.post("/generate/t2v", response_model=GenerationResponse)
async def generate_text_to_video(request: TextToVideoRequest):
    """
    Generate a video from a text prompt (Text-to-Video)
    """
    if model_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    job_id = str(uuid.uuid4())[:8]
    
    try:
        # Convert duration to valid frame count
        num_frames = duration_to_frames(request.duration)
        actual_duration = frames_to_duration(num_frames)
        
        output_path, gen_time, metadata = generate_video_internal(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            num_frames=num_frames,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            seed=request.seed,
            fps=LTX2_FPS,
        )
        
        filename = Path(output_path).name
        return GenerationResponse(
            status="success",
            job_id=filename.replace(".mp4", ""),
            video_url=f"/download/{filename}",
            generation_time_seconds=round(gen_time, 2),
            duration_seconds=actual_duration,
            num_frames=num_frames,
        )
        
    except Exception as e:
        traceback.print_exc()
        return GenerationResponse(
            status="error",
            job_id=job_id,
            message=str(e),
        )


@app.get("/download/{filename}")
async def download_video(filename: str):
    """Download a generated video by filename"""
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(str(file_path), media_type="video/mp4", filename=filename)


@app.delete("/videos/{job_id}")
async def delete_video(job_id: str):
    """Delete a generated video"""
    file_path = OUTPUT_DIR / f"{job_id}.mp4"
    if file_path.exists():
        file_path.unlink()
        return {"status": "deleted", "job_id": job_id}
    raise HTTPException(status_code=404, detail="Video not found")


@app.post("/reload")
async def reload_model(model_type: str = DEFAULT_MODEL_TYPE, profile: int = DEFAULT_PROFILE):
    """Reload the model (useful for switching model types)"""
    global current_model_type
    
    try:
        unload_model()
        load_wan2gp_model(model_type, profile)
        return {"status": "success", "model_type": current_model_type}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/info")
async def get_info():
    """Get API and model information"""
    return {
        "api_version": "2.0.0",
        "model_type": current_model_type,
        "model_loaded": model_instance is not None,
        "settings": {
            "fps": LTX2_FPS,
            "min_frames": LTX2_MIN_FRAMES,
            "frame_step": LTX2_FRAME_STEP,
            "resolution_divisor": LTX2_RESOLUTION_DIVISOR,
            "min_duration_seconds": round(LTX2_MIN_FRAMES / LTX2_FPS, 2),
            "max_duration_seconds": 20.0,
            "default_inference_steps": 8,
            "resolution_presets": {
                name: {"width": w, "height": h} 
                for name, (w, h) in RESOLUTION_PRESETS.items()
            },
        }
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # Arguments were already parsed at module load time (before wgp import)
    # Use the pre-parsed values
    
    print(f"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                     LTX-2 VIDEO GENERATION API                                ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║  Host:       {API_HOST:<60}
║  Port:       {API_PORT:<60}
║  Model:      {API_MODEL_TYPE:<60}
║  Profile:    {API_PROFILE:<60}
║  Output Dir: {str(OUTPUT_DIR):<60}
║  FPS:        {LTX2_FPS:<60}
╚═══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(
        "api_server:app",
        host=API_HOST,
        port=API_PORT,
        reload=API_RELOAD,
        workers=1,  # Single worker for GPU
    )


if __name__ == "__main__":
    main()
