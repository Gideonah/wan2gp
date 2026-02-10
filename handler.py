#!/usr/bin/env python3
"""
RunPod Serverless Handler for Wan2GP Video Generation

This handler follows RunPod's serverless pattern, loading the model once
at cold start and processing requests via the RunPod job queue.

Request Format (send to /run or /runsync):
{
    "input": {
        "prompt": "A person slowly turning and smiling",
        "image_url": "https://example.com/image.jpg",
        "duration": 5.0,
        "resolution_preset": "720p",
        "seed": -1
    }
}

Response Format:
{
    "output": {
        "status": "success",
        "video_url": "https://...",
        "generation_time_seconds": 45.2,
        ...
    }
}

Environment Variables:
    WAN2GP_MODEL_TYPE: Model to load (default: ltx2_distilled)
    WAN2GP_PROFILE: MMGP memory profile 1-6 (default: 3)
    WAN2GP_OUTPUT_DIR: Output directory (default: /workspace/outputs)
    GCS_ENABLED: Enable GCS upload (default: true)
    GCS_BUCKET_NAME: GCS bucket for video uploads
"""

import os
import sys
import time
import uuid
import gc
import traceback
import base64
import io
from pathlib import Path

# Add application root to path
APP_DIR = Path("/app")
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

# Set environment before imports
os.environ["GRADIO_LANG"] = "en"
os.environ["PYTHONUNBUFFERED"] = "1"

import runpod
import torch
import numpy as np
from PIL import Image
import httpx

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_MODEL_TYPE = os.environ.get("WAN2GP_MODEL_TYPE", "ltx2_distilled")
DEFAULT_PROFILE = int(os.environ.get("WAN2GP_PROFILE", "3"))
OUTPUT_DIR = Path(os.environ.get("WAN2GP_OUTPUT_DIR", "/workspace/outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# GCS Configuration
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "serverless_media_outputs")
GCS_ENABLED = os.environ.get("GCS_ENABLED", "false").lower() == "true"
GCS_PROJECT_ID = os.environ.get("GCP_PROJECT_ID", None)
GCS_URL_EXPIRATION_DAYS = int(os.environ.get("GCS_URL_EXPIRATION_DAYS", "7"))

# LTX-2 constants
LTX2_FPS = 24
LTX2_MIN_FRAMES = 17
LTX2_FRAME_STEP = 8

# Wan2.2 constants
WAN22_FPS = 16
WAN22_MIN_FRAMES = 5
WAN22_FRAME_STEP = 4

MODEL_FAMILIES = {
    "ltx2_distilled": "ltx2",
    "ltx2_19B": "ltx2",
    "z_image": "z_image",
    "i2v_2_2": "wan22",
    "i2v_2_2_Enhanced_Lightning_v2": "wan22",
}

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
        "default": (832, 480),
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
        "default": (832, 480),
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
    },
}

# Global model state
model_instance = None
model_handler = None
model_def = None
current_model_type = None
current_model_family = None
current_base_model_type = None
offloadobj = None


# ═══════════════════════════════════════════════════════════════════════════════
# GCS UPLOAD
# ═══════════════════════════════════════════════════════════════════════════════

_gcs_client = None


def get_gcp_credentials():
    """Build GCP credentials from environment variables."""
    client_email = os.environ.get("GCP_CLIENT_EMAIL")
    private_key_b64 = os.environ.get("GCP_PRIVATE_KEY_B64")

    if not client_email or not private_key_b64:
        return None

    try:
        from google.oauth2 import service_account

        private_key = base64.b64decode(private_key_b64).decode("utf-8")
        private_key = private_key.replace("\\n", "\n")

        credentials_info = {
            "type": "service_account",
            "project_id": os.environ.get("GCP_PROJECT_ID", ""),
            "private_key_id": os.environ.get("GCP_PRIVATE_KEY_ID", ""),
            "private_key": private_key,
            "client_email": client_email,
            "client_id": os.environ.get("GCP_CLIENT_ID", ""),
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
        }

        return service_account.Credentials.from_service_account_info(credentials_info)
    except Exception as e:
        print(f"⚠️ Failed to build GCP credentials: {e}")
        return None


def get_gcs_client():
    """Get or create GCS client."""
    global _gcs_client
    if _gcs_client is None:
        try:
            from google.cloud import storage

            credentials = get_gcp_credentials()
            if credentials:
                _gcs_client = storage.Client(project=GCS_PROJECT_ID, credentials=credentials)
            else:
                _gcs_client = storage.Client(project=GCS_PROJECT_ID)
            print(f"✅ GCS client initialized for bucket: {GCS_BUCKET_NAME}")
        except Exception as e:
            print(f"⚠️ GCS client not available: {e}")
            return None
    return _gcs_client


def upload_to_gcs(local_path: str, gcs_filename: str = None, content_type: str = "video/mp4"):
    """Upload a file to GCS and return a signed URL."""
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

        filename = gcs_filename or local_path.name
        gcs_path = f"videos/{filename}"

        bucket = client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(gcs_path)
        blob.chunk_size = 8 * 1024 * 1024

        print(f"📤 Uploading to GCS: gs://{GCS_BUCKET_NAME}/{gcs_path}")
        blob.upload_from_filename(str(local_path), content_type=content_type, timeout=600)

        signed_url = blob.generate_signed_url(
            version="v4",
            expiration=timedelta(days=GCS_URL_EXPIRATION_DAYS),
            method="GET",
            response_type=content_type,
            response_disposition="inline",
        )

        print(f"✅ Uploaded to GCS")
        return True, signed_url, None

    except Exception as e:
        return False, None, f"GCS upload failed: {str(e)}"


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════


def duration_to_frames_ltx2(duration_seconds: float) -> int:
    """Convert duration to valid frame count for LTX-2 (17 + 8*n)."""
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
    """Convert duration to valid frame count for Wan2.2 (5 + 4*n)."""
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
    """Convert frame count to duration in seconds."""
    return round(num_frames / fps, 2)


def resolve_resolution(model_family: str, resolution_preset=None, width=None, height=None):
    """Resolve resolution from preset or explicit values."""
    presets = RESOLUTION_PRESETS.get(model_family, RESOLUTION_PRESETS["ltx2"])

    if resolution_preset and resolution_preset in presets:
        return presets[resolution_preset]

    if model_family == "wan22":
        divisor, default_w, default_h = 16, 848, 480
    elif model_family == "z_image":
        divisor, default_w, default_h = 64, 1024, 1024
    else:
        divisor, default_w, default_h = 64, 768, 512

    w = width if width is not None else default_w
    h = height if height is not None else default_h
    w = max(256, (w // divisor) * divisor)
    h = max(256, (h // divisor) * divisor)

    return w, h


def fetch_image_from_url(url: str, timeout: float = 60.0) -> Image.Image:
    """Fetch an image from a URL."""
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content))
        return image.convert("RGB")


def decode_base64_image(image_base64: str) -> Image.Image:
    """Decode a base64 string to PIL Image."""
    if "," in image_base64:
        image_base64 = image_base64.split(",")[1]
    image_bytes = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image_bytes))
    return image.convert("RGB")


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════════


def load_model(model_type: str = DEFAULT_MODEL_TYPE, profile: int = DEFAULT_PROFILE):
    """Load the Wan2GP model into VRAM."""
    global model_instance, model_handler, model_def, current_model_type
    global current_model_family, current_base_model_type, offloadobj

    print(f"⏳ Loading model: {model_type} (profile: {profile})...")
    start_time = time.time()

    from wgp import load_models, get_model_def, get_base_model_type, get_model_handler

    model_def = get_model_def(model_type)
    base_model_type = get_base_model_type(model_type)
    model_handler = get_model_handler(base_model_type)

    model_instance, offloadobj = load_models(model_type, override_profile=profile)
    current_model_type = model_type
    current_base_model_type = base_model_type
    current_model_family = MODEL_FAMILIES.get(model_type, "unknown")

    load_time = time.time() - start_time
    print(f"✅ Model loaded in {load_time:.1f}s (family: {current_model_family})")

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"   GPU: {props.name} ({props.total_memory // 1024 // 1024}MB)")

    return model_instance


# ═══════════════════════════════════════════════════════════════════════════════
# VIDEO GENERATION
# ═══════════════════════════════════════════════════════════════════════════════


def convert_pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert PIL Image to tensor in range [-1, 1]."""
    return torch.from_numpy(np.array(image).astype(np.float32)).div_(127.5).sub_(1.0).movedim(-1, 0)


def generate_video(
    prompt: str,
    image_start: Image.Image = None,
    negative_prompt: str = "",
    width: int = 768,
    height: int = 512,
    num_frames: int = 121,
    num_inference_steps: int = 8,
    guidance_scale: float = 4.0,
    guidance2_scale: float = None,
    guidance_phases: int = 1,
    model_switch_phase: int = 1,
    switch_threshold: int = 0,
    flow_shift: float = None,
    seed: int = -1,
    fps: int = 24,
):
    """Generate video using the loaded model."""
    global model_instance, current_model_family, current_base_model_type, offloadobj

    from mmgp import offload
    from shared.utils.audio_video import save_video

    if model_instance is None:
        raise RuntimeError("Model not loaded")

    if seed < 0:
        seed = int(torch.randint(0, 2**32 - 1, (1,)).item())

    job_id = str(uuid.uuid4())[:8]
    output_path = OUTPUT_DIR / f"{job_id}.mp4"

    print(f"🎬 Generating video: {prompt[:50]}...")
    print(f"   Resolution: {width}x{height}, Frames: {num_frames}, Steps: {num_inference_steps}")

    start_time = time.time()

    # Set up offload shared state
    offload.shared_state["_attention"] = "sdpa"
    offload.shared_state["_chipmunk"] = False
    offload.shared_state["_radial"] = False
    offload.shared_state["_nag_scale"] = 1.0

    model_instance._interrupt = False

    loras_slists = {
        "phase1": [],
        "phase2": [],
        "phase3": [],
        "shared": [],
        "model_switch_step": num_inference_steps,
        "model_switch_step2": num_inference_steps,
    }

    def progress_callback(step, latents=None, force_update=False, override_num_inference_steps=None, **kwargs):
        if step >= 0:
            steps_display = override_num_inference_steps or num_inference_steps
            print(f"   Step {step + 1}/{steps_display}")

    # Convert image for Wan2.2
    image_start_tensor = None
    input_video_tensor = None
    if image_start is not None and current_model_family == "wan22":
        image_start_tensor = convert_pil_to_tensor(image_start)
        input_video_tensor = image_start_tensor.unsqueeze(1)

    gen_kwargs = {
        "input_prompt": prompt,
        "n_prompt": negative_prompt if negative_prompt else None,
        "image_start": image_start_tensor,
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
        "callback": progress_callback,
        "model_type": current_base_model_type,
        "offloadobj": offloadobj,
        "set_header_text": lambda txt: print(f"   Phase: {txt}"),
    }

    if input_video_tensor is not None:
        gen_kwargs["input_video"] = input_video_tensor
    elif image_start is not None and current_model_family != "wan22":
        gen_kwargs["image_start"] = image_start

    if flow_shift is not None:
        gen_kwargs["shift"] = flow_shift

    if guidance_phases >= 1:
        gen_kwargs["guide_phases"] = guidance_phases
        gen_kwargs["model_switch_phase"] = model_switch_phase
        gen_kwargs["switch_threshold"] = switch_threshold
        if guidance2_scale is not None:
            gen_kwargs["guide2_scale"] = guidance2_scale

    if current_model_family == "wan22":
        gen_kwargs["sample_solver"] = "euler"

    try:
        # Initialize cache attribute
        if hasattr(model_instance, "model") and model_instance.model is not None:
            object.__setattr__(model_instance.model, "cache", None)
        if hasattr(model_instance, "model2") and model_instance.model2 is not None:
            object.__setattr__(model_instance.model2, "cache", None)

        result = model_instance.generate(**gen_kwargs)

        if isinstance(result, dict):
            video_tensor = result.get("x", result)
            audio_data = result.get("audio", None)
            audio_sr = result.get("audio_sampling_rate", 48000)
        else:
            video_tensor = result
            audio_data = None
            audio_sr = 48000

        save_video(video_tensor, str(output_path), fps=fps)

        # Mux audio if present
        if audio_data is not None:
            from postprocessing.mmaudio.data.av_utils import remux_with_audio

            if isinstance(audio_data, np.ndarray):
                audio_tensor = torch.from_numpy(audio_data)
            else:
                audio_tensor = audio_data
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            elif audio_tensor.dim() == 2:
                if audio_tensor.shape[0] > audio_tensor.shape[1] and audio_tensor.shape[1] in (1, 2):
                    audio_tensor = audio_tensor.T
            temp_video_path = output_path.with_name(output_path.stem + "_tmp.mp4")
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

    return str(output_path), generation_time, {
        "num_frames": num_frames,
        "duration": frames_to_duration(num_frames, fps),
        "fps": fps,
        "width": width,
        "height": height,
        "seed": seed,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# RUNPOD HANDLER
# ═══════════════════════════════════════════════════════════════════════════════


def handler(job):
    """
    RunPod serverless handler function.

    Input format:
    {
        "input": {
            "prompt": "A person slowly turning and smiling",
            "image_url": "https://example.com/image.jpg",
            "duration": 5.0,
            "resolution_preset": "720p",
            "seed": -1,
            ...
        }
    }
    """
    try:
        job_input = job.get("input", {})

        # Extract parameters
        prompt = job_input.get("prompt", "")
        image_url = job_input.get("image_url")
        image_base64 = job_input.get("image_base64")
        duration = job_input.get("duration", 5.0)
        resolution_preset = job_input.get("resolution_preset")
        width = job_input.get("width")
        height = job_input.get("height")
        seed = job_input.get("seed", -1)
        guidance_scale = job_input.get("guidance_scale", 4.0)
        num_inference_steps = job_input.get("num_inference_steps")

        print(f"📥 Received job: prompt={prompt[:50]}...")

        # Determine parameters based on model family
        if current_model_family == "wan22":
            fps = WAN22_FPS
            num_frames = duration_to_frames_wan22(duration)
            num_inference_steps = num_inference_steps or 4
            guidance2_scale = job_input.get("guidance2_scale", 1.0)
            guidance_phases = job_input.get("guidance_phases", 2)
            model_switch_phase = job_input.get("model_switch_phase", 1)
            switch_threshold = job_input.get("switch_threshold", 900)
            flow_shift = job_input.get("flow_shift", 5.0)
        else:  # ltx2
            fps = LTX2_FPS
            num_frames = duration_to_frames_ltx2(duration)
            num_inference_steps = num_inference_steps or 8
            guidance2_scale = None
            guidance_phases = 1
            model_switch_phase = 1
            switch_threshold = 0
            flow_shift = None

        # Resolve resolution
        w, h = resolve_resolution(current_model_family, resolution_preset, width, height)

        # Fetch or decode image
        image_start = None
        if image_url:
            print(f"📥 Fetching image from: {image_url[:80]}...")
            image_start = fetch_image_from_url(image_url)
            image_start = image_start.resize((w, h), Image.LANCZOS)
        elif image_base64:
            print(f"📥 Decoding base64 image...")
            image_start = decode_base64_image(image_base64)
            image_start = image_start.resize((w, h), Image.LANCZOS)

        # Generate video
        output_path, gen_time, metadata = generate_video(
            prompt=prompt,
            image_start=image_start,
            width=w,
            height=h,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            guidance2_scale=guidance2_scale,
            guidance_phases=guidance_phases,
            model_switch_phase=model_switch_phase,
            switch_threshold=switch_threshold,
            flow_shift=flow_shift,
            seed=seed,
            fps=fps,
        )

        # Upload to GCS and return signed URL
        filename = Path(output_path).name
        gcs_success, gcs_url, gcs_error = upload_to_gcs(output_path, filename)

        if gcs_success:
            video_url = gcs_url
        else:
            print(f"⚠️ GCS upload failed: {gcs_error}")
            # Return local path as fallback if GCS fails
            video_url = str(output_path)

        return {
            "status": "success",
            "job_id": filename.replace(".mp4", ""),
            "video_url": video_url,
            "generation_time_seconds": round(gen_time, 2),
            "duration_seconds": metadata["duration"],
            "num_frames": metadata["num_frames"],
            "width": metadata["width"],
            "height": metadata["height"],
            "seed": metadata["seed"],
            "model_type": current_model_type,
            "model_family": current_model_family,
        }

    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

print("═══════════════════════════════════════════════════════════════════")
print("  Wan2GP RunPod Serverless Handler")
print("═══════════════════════════════════════════════════════════════════")
print(f"  Model Type: {DEFAULT_MODEL_TYPE}")
print(f"  Profile:    {DEFAULT_PROFILE}")
print(f"  Output Dir: {OUTPUT_DIR}")
print(f"  GCS Enabled: {GCS_ENABLED}")
if GCS_ENABLED:
    print(f"  GCS Bucket: {GCS_BUCKET_NAME}")
print("═══════════════════════════════════════════════════════════════════")

# Load model at cold start (before accepting requests)
print("\n🚀 Loading model (cold start)...")
load_model(DEFAULT_MODEL_TYPE, DEFAULT_PROFILE)

# Start RunPod serverless handler
print("\n✅ Ready to accept requests")
runpod.serverless.start({"handler": handler})


