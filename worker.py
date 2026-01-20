#!/usr/bin/env python3
"""
Vast.ai PyWorker for LTX-2 Distilled Video Generation

This worker.py configures the Vast serverless proxy to route requests
to the LTX-2 FastAPI server (api_server.py).

The PyWorker:
  - Proxies /generate/i2v (image URL), /generate/i2v-base64, and /generate/t2v
  - Monitors logs for model readiness
  - Runs benchmarks to measure throughput
  - Reports workload metrics for autoscaling

Environment Variables:
  MODEL_SERVER_PORT: Port where api_server.py runs (default: 8000)
  WAN2GP_LOG_FILE: Log file path (default: /var/log/wan2gp/server.log)
"""

import os
import random

from vastai import (
    Worker,
    WorkerConfig,
    HandlerConfig,
    BenchmarkConfig,
    LogActionConfig,
)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_SERVER_URL = "http://127.0.0.1"
MODEL_SERVER_PORT = int(os.environ.get("MODEL_SERVER_PORT", "8000"))
MODEL_LOG_FILE = os.environ.get("WAN2GP_LOG_FILE", "/var/log/wan2gp/server.log")

# LTX-2 specific settings
LTX2_FPS = 24
LTX2_MIN_FRAMES = 17
LTX2_FRAME_STEP = 8

# ═══════════════════════════════════════════════════════════════════════════════
# LOG ACTION PATTERNS
# ═══════════════════════════════════════════════════════════════════════════════

# These patterns detect model state from log output
# Prefix-based matching (case-sensitive)

MODEL_LOAD_PATTERNS = [
    # Uvicorn startup complete
    "Application startup complete",
    # Our custom log from api_server.py
    "✅ Model loaded",
    # Alternative pattern
    "Uvicorn running on",
]

MODEL_ERROR_PATTERNS = [
    # Model loading failure
    "❌ Failed to load model",
    # Python exceptions
    "Traceback (most recent call last):",
    "RuntimeError:",
    "CUDA out of memory",
    "torch.cuda.OutOfMemoryError",
    # Process crashes
    "Segmentation fault",
    "killed",
]

MODEL_INFO_PATTERNS = [
    # Download progress
    "Downloading",
    # Model loading stages
    "Loading",
    "⏳",
]


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def duration_to_frames(duration_seconds: float) -> int:
    """Convert duration in seconds to valid frame count for LTX-2."""
    target_frames = int(duration_seconds * LTX2_FPS)
    if target_frames < LTX2_MIN_FRAMES:
        return LTX2_MIN_FRAMES
    n = max(0, (target_frames - LTX2_MIN_FRAMES) // LTX2_FRAME_STEP)
    return LTX2_MIN_FRAMES + (n * LTX2_FRAME_STEP)


# ═══════════════════════════════════════════════════════════════════════════════
# WORKLOAD CALCULATION
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_video_workload(payload: dict) -> float:
    """
    Calculate workload for a video generation request.
    
    Workload is proportional to:
      - Duration/frames (more frames = more work)
      - Resolution (more pixels = more work)
      - Inference steps (more steps = more work)
    
    This metric is used by Vast's autoscaler to right-size capacity.
    """
    # Extract parameters with LTX-2 defaults
    duration = payload.get("duration", 5.0)
    num_frames = payload.get("num_frames", duration_to_frames(duration))
    width = payload.get("width", 768)
    height = payload.get("height", 512)
    num_inference_steps = payload.get("num_inference_steps", 8)  # LTX-2 distilled uses 8 steps
    
    # Normalize to a reasonable scale
    # Base workload: 121 frames @ 768x512 @ 8 steps = 1000 units
    base_frames = 121
    base_pixels = 768 * 512
    base_steps = 8
    base_workload = 1000.0
    
    # Calculate relative workload
    frame_factor = num_frames / base_frames
    pixel_factor = (width * height) / base_pixels
    step_factor = num_inference_steps / base_steps
    
    workload = base_workload * frame_factor * pixel_factor * step_factor
    
    return workload


# ═══════════════════════════════════════════════════════════════════════════════
# BENCHMARK PAYLOAD GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

# Sample prompts for benchmarking
BENCHMARK_PROMPTS = [
    "A serene mountain lake at sunset with golden light reflecting on the water",
    "A person slowly turning their head and smiling warmly at the camera",
    "Ocean waves gently crashing on a sandy beach with palm trees swaying",
]

# Sample image URLs for benchmarking (use reliable, fast-loading images)
BENCHMARK_IMAGE_URLS = [
    "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=512&h=512&fit=crop",
    "https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=512&h=512&fit=crop",
]


def i2v_benchmark_generator() -> dict:
    """
    Generate a benchmark payload for /generate/i2v endpoint.
    
    Uses smaller parameters for faster benchmark completion while
    still exercising the full pipeline.
    """
    prompt = random.choice(BENCHMARK_PROMPTS)
    image_url = random.choice(BENCHMARK_IMAGE_URLS)
    
    return {
        "prompt": prompt,
        "image_url": image_url,
        "duration": 1.0,        # ~1 second = 25 frames (smallest valid for LTX-2)
        "width": 512,           # Smaller for faster benchmark
        "height": 512,          # Square aspect ratio
        "guidance_scale": 4.0,
        "seed": random.randint(0, 2**31 - 1),
    }


def t2v_benchmark_generator() -> dict:
    """
    Generate a benchmark payload for /generate/t2v endpoint.
    """
    prompt = random.choice(BENCHMARK_PROMPTS)
    
    return {
        "prompt": prompt,
        "negative_prompt": "blurry, low quality, distorted",
        "duration": 1.0,        # ~1 second video
        "width": 512,           # Smaller for faster benchmark
        "height": 512,          # Square aspect ratio
        "num_inference_steps": 8,
        "guidance_scale": 4.0,
        "seed": random.randint(0, 2**31 - 1),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# HANDLER CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Image-to-Video handler (URL input) - Primary endpoint with benchmarking
i2v_handler = HandlerConfig(
    route="/generate/i2v",
    
    # Video generation is GPU-bound, process one at a time
    allow_parallel_requests=False,
    
    # Video generation can take several minutes
    # Allow 10 minutes queue time before 429
    max_queue_time=600.0,
    
    # Workload calculation for autoscaling
    workload_calculator=calculate_video_workload,
    
    # Benchmark configuration
    benchmark_config=BenchmarkConfig(
        generator=i2v_benchmark_generator,
        runs=1,          # Single run since video gen is slow
        concurrency=1,   # Serial execution (GPU-bound)
    ),
)

# Image-to-Video handler (base64 input) - Alternative endpoint
i2v_base64_handler = HandlerConfig(
    route="/generate/i2v-base64",
    
    # Video generation is GPU-bound
    allow_parallel_requests=False,
    
    # Allow 10 minutes queue time
    max_queue_time=600.0,
    
    # Same workload calculation
    workload_calculator=calculate_video_workload,
)

# Text-to-Video handler
t2v_handler = HandlerConfig(
    route="/generate/t2v",
    
    # Video generation is GPU-bound
    allow_parallel_requests=False,
    
    # Allow 10 minutes queue time
    max_queue_time=600.0,
    
    # Same workload calculation
    workload_calculator=calculate_video_workload,
)

# Health check handler (simple pass-through)
health_handler = HandlerConfig(
    route="/health",
    
    # Health checks are lightweight
    allow_parallel_requests=True,
    
    # Short timeout
    max_queue_time=10.0,
    
    # Constant minimal workload
    workload_calculator=lambda payload: 1.0,
)

# Root health check (same as /health)
root_handler = HandlerConfig(
    route="/",
    allow_parallel_requests=True,
    max_queue_time=10.0,
    workload_calculator=lambda payload: 1.0,
)

# Info endpoint
info_handler = HandlerConfig(
    route="/info",
    allow_parallel_requests=True,
    max_queue_time=10.0,
    workload_calculator=lambda payload: 1.0,
)


# ═══════════════════════════════════════════════════════════════════════════════
# WORKER CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

worker_config = WorkerConfig(
    # Backend server connection
    model_server_url=MODEL_SERVER_URL,
    model_server_port=MODEL_SERVER_PORT,
    model_log_file=MODEL_LOG_FILE,
    
    # Route handlers
    handlers=[
        i2v_handler,
        i2v_base64_handler,
        t2v_handler,
        health_handler,
        root_handler,
        info_handler,
    ],
    
    # Log-based state detection
    log_action_config=LogActionConfig(
        on_load=MODEL_LOAD_PATTERNS,
        on_error=MODEL_ERROR_PATTERNS,
        on_info=MODEL_INFO_PATTERNS,
    ),
)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("═══════════════════════════════════════════════════════════════════")
    print("  LTX-2 Distilled PyWorker for Vast.ai Serverless")
    print("═══════════════════════════════════════════════════════════════════")
    print(f"  Model Server: {MODEL_SERVER_URL}:{MODEL_SERVER_PORT}")
    print(f"  Log File:     {MODEL_LOG_FILE}")
    print(f"  Routes:")
    print(f"    - /generate/i2v       (Image URL → Video)")
    print(f"    - /generate/i2v-base64 (Base64 Image → Video)")
    print(f"    - /generate/t2v       (Text → Video)")
    print(f"    - /health, /info")
    print("═══════════════════════════════════════════════════════════════════")
    print("")
    
    # Start the PyWorker
    Worker(worker_config).run()
