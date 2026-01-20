#!/usr/bin/env python3
"""
Vast.ai PyWorker for Wan2GP Video Generation

This worker.py configures the Vast serverless proxy to route requests
to the Wan2GP FastAPI server (api_server.py).

The PyWorker:
  - Proxies /generate/t2v and /generate/i2v to the backend
  - Monitors logs for model readiness
  - Runs benchmarks to measure throughput
  - Reports workload metrics for autoscaling

Environment Variables:
  MODEL_SERVER_PORT: Port where api_server.py runs (default: 8000)
  WAN2GP_LOG_FILE: Log file path (default: /var/log/wan2gp/server.log)
"""

import os
import random
import string

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
# WORKLOAD CALCULATION
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_video_workload(payload: dict) -> float:
    """
    Calculate workload for a video generation request.
    
    Workload is proportional to:
      - Number of frames (more frames = more work)
      - Resolution (more pixels = more work)
      - Inference steps (more steps = more work)
    
    This metric is used by Vast's autoscaler to right-size capacity.
    """
    # Extract parameters with defaults matching api_server.py
    num_frames = payload.get("num_frames", 81)
    width = payload.get("width", 832)
    height = payload.get("height", 480)
    num_inference_steps = payload.get("num_inference_steps", 30)
    
    # Normalize to a reasonable scale
    # Base workload: 81 frames @ 832x480 @ 30 steps = 1000 units
    base_frames = 81
    base_pixels = 832 * 480
    base_steps = 30
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

]


def t2v_benchmark_generator() -> dict:
    """
    Generate a benchmark payload for /generate/t2v endpoint.
    
    Uses smaller parameters for faster benchmark completion while
    still exercising the full pipeline.
    """
    prompt = random.choice(BENCHMARK_PROMPTS)
    
    return {
        "prompt": prompt,
        "negative_prompt": "blurry, low quality, distorted",
        "width": 512,           # Smaller for faster benchmark
        "height": 320,          # Smaller for faster benchmark
        "num_frames": 17,       # Minimal frames (1 second at 16fps)
        "num_inference_steps": 20,  # Fewer steps for speed
        "guidance_scale": 5.0,
        "flow_shift": 5.0,
        "seed": random.randint(0, 2**31 - 1),
        "sample_solver": "unipc",
        "fps": 16,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# HANDLER CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Text-to-Video handler (with benchmarking)
t2v_handler = HandlerConfig(
    route="/generate/t2v",
    
    # Video generation is GPU-bound, process one at a time
    allow_parallel_requests=False,
    
    # Video generation can take several minutes
    # Allow 10 minutes queue time before 429
    max_queue_time=600.0,
    
    # Workload calculation for autoscaling
    workload_calculator=calculate_video_workload,
    
    # Benchmark configuration (only one handler should have this)
    benchmark_config=BenchmarkConfig(
        generator=t2v_benchmark_generator,
        runs=1,          # Only 2 runs since video gen is slow
        concurrency=1,   # Serial execution (GPU-bound)
    ),
)

# Image-to-Video handler (no benchmark - uses same model)
i2v_handler = HandlerConfig(
    route="/generate/i2v",
    
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
        t2v_handler,
        i2v_handler,
        health_handler,
        root_handler,
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
    print("  Wan2GP PyWorker for Vast.ai Serverless")
    print("═══════════════════════════════════════════════════════════════════")
    print(f"  Model Server: {MODEL_SERVER_URL}:{MODEL_SERVER_PORT}")
    print(f"  Log File:     {MODEL_LOG_FILE}")
    print(f"  Routes:       /generate/t2v, /generate/i2v, /health")
    print("═══════════════════════════════════════════════════════════════════")
    print("")
    
    # Start the PyWorker
    Worker(worker_config).run()

