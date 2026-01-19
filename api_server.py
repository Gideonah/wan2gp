#!/usr/bin/env python3
"""
Wan2GP API Server for Serverless Deployment (Vast.ai, RunPod, Modal)

This FastAPI server provides a REST API interface to Wan2GP's video generation
capabilities, designed for serverless GPU deployments.

Usage:
    python api_server.py [--port 8000] [--model-type t2v] [--profile 5]

Environment Variables:
    WAN2GP_MODEL_TYPE: Default model type to load (e.g., "t2v", "i2v", "vace_14B")
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
    parser.add_argument("--model-type", type=str, default=os.environ.get("WAN2GP_MODEL_TYPE", "t2v"))
    parser.add_argument("--profile", type=int, default=int(os.environ.get("WAN2GP_PROFILE", "5")))
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("-h", "--help", action="store_true")
    
    args, unknown = parser.parse_known_args()
    
    if args.help:
        print("""
Wan2GP API Server

Usage:
    python api_server.py [OPTIONS]

Options:
    --host          Host to bind to (default: 0.0.0.0)
    --port          Port to listen on (default: 8000)
    --model-type    Model type to load: t2v, i2v, vace_14B, etc. (default: t2v)
    --profile       MMGP memory profile 1-6 (default: 5)
    --reload        Enable auto-reload for development

Environment Variables:
    WAN2GP_MODEL_TYPE   Default model type
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

# Global model reference
model_instance = None
model_handler = None
model_def = None
current_model_type = None
offloadobj = None

# ═══════════════════════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class TextToVideoRequest(BaseModel):
    """Request model for text-to-video generation"""
    prompt: str = Field(..., description="Text prompt describing the video")
    negative_prompt: str = Field("", description="Negative prompt")
    width: int = Field(832, description="Video width (must be multiple of 16)")
    height: int = Field(480, description="Video height (must be multiple of 16)")
    num_frames: int = Field(81, description="Number of frames (5, 9, 13... up to 121)")
    num_inference_steps: int = Field(30, description="Number of denoising steps")
    guidance_scale: float = Field(5.0, description="Classifier-free guidance scale")
    flow_shift: float = Field(5.0, description="Flow matching shift parameter")
    seed: int = Field(-1, description="Random seed (-1 for random)")
    sample_solver: str = Field("unipc", description="Sampler: unipc, euler, dpm++")
    fps: int = Field(16, description="Output video FPS")

class ImageToVideoRequest(BaseModel):
    """Request model for image-to-video generation"""
    prompt: str = Field(..., description="Text prompt describing the video")
    image_base64: str = Field(..., description="Base64 encoded start image (PNG/JPEG)")
    negative_prompt: str = Field("", description="Negative prompt")
    width: int = Field(832, description="Video width")
    height: int = Field(480, description="Video height")
    num_frames: int = Field(81, description="Number of frames")
    num_inference_steps: int = Field(30, description="Number of denoising steps")
    guidance_scale: float = Field(5.0, description="Classifier-free guidance scale")
    flow_shift: float = Field(5.0, description="Flow matching shift parameter")
    seed: int = Field(-1, description="Random seed (-1 for random)")
    sample_solver: str = Field("unipc", description="Sampler: unipc, euler, dpm++")
    fps: int = Field(16, description="Output video FPS")

class GenerationResponse(BaseModel):
    """Response model for video generation"""
    status: str
    job_id: str
    message: Optional[str] = None
    video_url: Optional[str] = None
    generation_time_seconds: Optional[float] = None

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_type: Optional[str] = None
    gpu_name: Optional[str] = None
    gpu_memory_total_mb: Optional[int] = None
    gpu_memory_used_mb: Optional[int] = None

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_wan2gp_model(model_type: str = DEFAULT_MODEL_TYPE, profile: int = DEFAULT_PROFILE):
    """
    Load the Wan2GP model into VRAM.
    This is called once at startup to keep the model warm.
    """
    global model_instance, model_handler, model_def, current_model_type, offloadobj
    
    print(f"⏳ Loading Wan2GP model: {model_type} (profile: {profile})...")
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

def decode_base64_image(image_base64: str) -> Image.Image:
    """Decode a base64 string to PIL Image"""
    # Handle data URL format
    if "," in image_base64:
        image_base64 = image_base64.split(",")[1]
    
    image_bytes = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image_bytes))
    return image.convert("RGB")

def generate_video_internal(
    prompt: str,
    image_start: Optional[Image.Image] = None,
    negative_prompt: str = "",
    width: int = 832,
    height: int = 480,
    num_frames: int = 81,
    num_inference_steps: int = 30,
    guidance_scale: float = 5.0,
    flow_shift: float = 5.0,
    seed: int = -1,
    sample_solver: str = "unipc",
    fps: int = 16,
) -> tuple[str, float]:
    """
    Internal function to generate video using the loaded model.
    Returns (output_path, generation_time)
    """
    global model_instance, model_def, current_model_type
    
    if model_instance is None:
        raise RuntimeError("Model not loaded")
    
    # Import wgp utilities
    from wgp import get_transformer_model
    
    # Validate dimensions
    width = (width // 16) * 16
    height = (height // 16) * 16
    
    # Handle seed
    if seed < 0:
        seed = int(torch.randint(0, 2**32 - 1, (1,)).item())
    
    # Create job ID and output path
    job_id = str(uuid.uuid4())[:8]
    output_path = OUTPUT_DIR / f"{job_id}.mp4"
    
    print(f"🎬 Generating video: {prompt[:50]}...")
    print(f"   Resolution: {width}x{height}, Frames: {num_frames}, Steps: {num_inference_steps}")
    
    start_time = time.time()
    
    # Prepare image tensor if provided
    input_video = None
    if image_start is not None:
        from shared.utils.utils import convert_image_to_tensor
        # Resize image to target dimensions
        image_start = image_start.resize((width, height), Image.LANCZOS)
        input_video = convert_image_to_tensor(image_start).unsqueeze(1)
    
    # Set up transformer cache (required by the model)
    trans = get_transformer_model(model_instance)
    trans.cache = None  # Disable step caching for simplicity
    trans2 = get_transformer_model(model_instance, 2)
    if trans2 is not None:
        trans2.cache = None
    
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
    
    # Dummy callback function (required by generate)
    # Signature must match: callback(step, latents, force_update, show_latents, **kwargs)
    def progress_callback(*args, **kwargs):
        step = args[0] if len(args) > 0 else -1
        if step >= 0:
            print(f"   Step {step + 1}/{num_inference_steps}")
    
    # Run generation
    try:
        result = model_instance.generate(
            input_prompt=prompt,
            n_prompt=negative_prompt,
            input_video=input_video,
            width=width,
            height=height,
            frame_num=num_frames,
            sampling_steps=num_inference_steps,
            guide_scale=guidance_scale,
            shift=flow_shift,
            seed=seed,
            sample_solver=sample_solver,
            callback=progress_callback,
            VAE_tile_size=0,  # Auto-detect
            model_type=current_model_type,
            loras_slists=loras_slists,
        )
        
        # Extract video tensor
        if isinstance(result, dict):
            video_tensor = result.get("x", result)
        else:
            video_tensor = result
        
        # Save video
        save_video(video_tensor, str(output_path), fps=fps)
        
    except Exception as e:
        raise RuntimeError(f"Generation failed: {str(e)}")
    
    finally:
        # Cleanup
        gc.collect()
        torch.cuda.empty_cache()
    
    generation_time = time.time() - start_time
    print(f"✅ Video saved to {output_path} in {generation_time:.1f}s")
    
    return str(output_path), generation_time

# ═══════════════════════════════════════════════════════════════════════════════
# FASTAPI APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan handler for model loading/unloading"""
    # Startup: Load model
    print("🚀 Starting Wan2GP API Server...")
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
    title="Wan2GP API",
    description="REST API for Wan2GP video generation",
    version="1.0.0",
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

@app.post("/generate/t2v", response_model=GenerationResponse)
async def generate_text_to_video(request: TextToVideoRequest):
    """
    Generate a video from a text prompt (Text-to-Video)
    """
    if model_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    job_id = str(uuid.uuid4())[:8]
    
    try:
        output_path, gen_time = generate_video_internal(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            num_frames=request.num_frames,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            flow_shift=request.flow_shift,
            seed=request.seed,
            sample_solver=request.sample_solver,
            fps=request.fps,
        )
        
        filename = Path(output_path).name
        return GenerationResponse(
            status="success",
            job_id=filename.replace(".mp4", ""),
            video_url=f"/download/{filename}",
            generation_time_seconds=round(gen_time, 2),
        )
        
    except Exception as e:
        traceback.print_exc()
        return GenerationResponse(
            status="error",
            job_id=job_id,
            message=str(e),
        )

@app.post("/generate/i2v", response_model=GenerationResponse)
async def generate_image_to_video(request: ImageToVideoRequest):
    """
    Generate a video from an image and text prompt (Image-to-Video)
    """
    if model_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    job_id = str(uuid.uuid4())[:8]
    
    try:
        # Decode the input image
        image_start = decode_base64_image(request.image_base64)
        
        output_path, gen_time = generate_video_internal(
            prompt=request.prompt,
            image_start=image_start,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            num_frames=request.num_frames,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            flow_shift=request.flow_shift,
            seed=request.seed,
            sample_solver=request.sample_solver,
            fps=request.fps,
        )
        
        filename = Path(output_path).name
        return GenerationResponse(
            status="success",
            job_id=filename.replace(".mp4", ""),
            video_url=f"/download/{filename}",
            generation_time_seconds=round(gen_time, 2),
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

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # Arguments were already parsed at module load time (before wgp import)
    # Use the pre-parsed values
    
    print(f"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                        WAN2GP API SERVER                                      ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║  Host:       {API_HOST}                                                       
║  Port:       {API_PORT}                                                       
║  Model:      {API_MODEL_TYPE}                                                 
║  Profile:    {API_PROFILE}                                                    
║  Output Dir: {OUTPUT_DIR}                                                      
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

