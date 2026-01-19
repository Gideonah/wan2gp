# Wan2GP API Deployment Guide

This guide explains how to deploy Wan2GP as a serverless API on platforms like **Vast.ai**, **RunPod**, or **Modal**.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Docker Container                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────┐    ┌─────────────────────────────────┐   │
│  │  entrypoint_api  │───▶│  api_server.py (FastAPI)        │   │
│  └──────────────────┘    │                                  │   │
│                           │  /health          - Status      │   │
│                           │  /generate/t2v    - Text→Video  │   │
│                           │  /generate/i2v    - Image→Video │   │
│                           │  /download/{id}   - Get Video   │   │
│                           └─────────────────────────────────┘   │
│                                        │                         │
│                                        ▼                         │
│                           ┌─────────────────────────────────┐   │
│                           │  WanAny2V Model (in VRAM)       │   │
│                           │  - Loaded once at startup       │   │
│                           │  - Stays warm between requests  │   │
│                           └─────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Files Created

| File | Purpose |
|------|---------|
| `api_server.py` | FastAPI server that loads model and handles requests |
| `entrypoint_api.sh` | Docker entrypoint for API mode |
| `Dockerfile.api` | Docker image optimized for API deployment |
| `api_client.py` | Python client example for testing |

## Quick Start

### 1. Build the Docker Image

```bash
# Build for common GPU architectures (RTX 3000/4000 series)
docker build -f Dockerfile.api -t wan2gp-api .

# Build for specific GPU (e.g., H100)
docker build -f Dockerfile.api \
  --build-arg CUDA_ARCHITECTURES="9.0" \
  -t wan2gp-api:h100 .
```

### 2. Run Locally

```bash
docker run --gpus all -p 8000:8000 \
  -v /path/to/models:/workspace/ckpts \
  -e WAN2GP_MODEL_TYPE=t2v \
  -e WAN2GP_PROFILE=5 \
  wan2gp-api
```

### 3. Test the API

```bash
# Check health
curl http://localhost:8000/health

# Generate a video (Text-to-Video)
curl -X POST http://localhost:8000/generate/t2v \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A beautiful sunset over the ocean, waves crashing",
    "width": 832,
    "height": 480,
    "num_frames": 81,
    "num_inference_steps": 30
  }'
```

## Deployment on Vast.ai

### Option A: Always-On Instance

1. **Select a GPU** (RTX 4090, A100, H100 recommended)

2. **Docker Configuration**:
   ```
   Image: your-registry/wan2gp-api:latest
   Docker Options: --gpus all
   Port Mappings: 8000:8000 (TCP)
   ```

3. **Environment Variables**:
   ```
   WAN2GP_MODEL_TYPE=t2v
   WAN2GP_PROFILE=5
   ```

4. **Launch and note the external IP**

5. **Access the API**:
   ```bash
   curl http://VAST_IP:PORT/health
   ```

### Option B: Pre-baked Weights (Recommended)

To avoid downloading 14GB+ weights on each startup:

1. **Rent a machine with sufficient storage**

2. **Download weights inside the container**:
   ```bash
   # SSH into the instance
   cd /workspace
   # Run the model once to trigger download
   python -c "from wgp import load_models; load_models('t2v')"
   ```

3. **Commit the Docker image**:
   ```bash
   docker commit CONTAINER_ID your-registry/wan2gp-api:with-weights
   docker push your-registry/wan2gp-api:with-weights
   ```

4. **Use the new image** for future deployments

## API Reference

### Health Check

```http
GET /health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_type": "t2v",
  "gpu_name": "NVIDIA GeForce RTX 4090",
  "gpu_memory_total_mb": 24564,
  "gpu_memory_used_mb": 18432
}
```

### Text-to-Video

```http
POST /generate/t2v
Content-Type: application/json

{
  "prompt": "A cat walking in a garden",
  "negative_prompt": "blurry, low quality",
  "width": 832,
  "height": 480,
  "num_frames": 81,
  "num_inference_steps": 30,
  "guidance_scale": 5.0,
  "flow_shift": 5.0,
  "seed": -1,
  "sample_solver": "unipc",
  "fps": 16
}
```

Response:
```json
{
  "status": "success",
  "job_id": "a1b2c3d4",
  "video_url": "/download/a1b2c3d4.mp4",
  "generation_time_seconds": 45.2
}
```

### Image-to-Video

```http
POST /generate/i2v
Content-Type: application/json

{
  "prompt": "Person dancing gracefully",
  "image_base64": "base64_encoded_image_data...",
  "width": 832,
  "height": 480,
  "num_frames": 81,
  "num_inference_steps": 30,
  "guidance_scale": 5.0,
  "seed": -1
}
```

### Download Video

```http
GET /download/{filename}
```

Returns the video file directly (application/mp4).

### Delete Video

```http
DELETE /videos/{job_id}
```

### Reload Model

```http
POST /reload?model_type=i2v&profile=5
```

Use this to switch between model types without restarting the container.

## Python Client Example

```python
from api_client import Wan2GPClient

client = Wan2GPClient("http://YOUR_SERVER:8000")

# Check health
health = client.health_check()
print(f"Model: {health['model_type']}, GPU: {health['gpu_name']}")

# Generate text-to-video
result = client.text_to_video(
    prompt="A majestic eagle soaring over mountains",
    width=832,
    height=480,
    num_frames=81,
    num_inference_steps=30,
)

# Download the result
if result['status'] == 'success':
    client.download_video(result['video_url'], "eagle.mp4")
    print(f"Video saved! Generation took {result['generation_time_seconds']}s")
```

## Configuration Options

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WAN2GP_MODEL_TYPE` | `t2v` | Model type: `t2v`, `i2v`, `vace_14B`, etc. |
| `WAN2GP_PROFILE` | `5` | MMGP memory profile (1-6, higher = more VRAM) |
| `WAN2GP_PORT` | `8000` | API server port |
| `WAN2GP_OUTPUT_DIR` | `/workspace/outputs` | Directory for generated videos |

### Memory Profiles

| Profile | Description | Recommended VRAM |
|---------|-------------|------------------|
| 1 | Minimal VRAM | 8GB |
| 2 | Low VRAM | 12GB |
| 3 | Medium VRAM | 16GB |
| 4 | High VRAM | 20GB |
| 5 | Very High VRAM | 24GB |
| 6 | Maximum | 32GB+ |

### Model Types

| Type | Description | Use Case |
|------|-------------|----------|
| `t2v` | Text-to-Video 14B | General video from text |
| `i2v` | Image-to-Video | Animate a starting image |
| `t2v_1.3B` | Text-to-Video 1.3B | Faster, lower VRAM |
| `vace_14B` | VACE Video | Control with poses/depth |
| `i2v_2_2` | I2V Wan 2.2 | Latest I2V model |

## Troubleshooting

### Model takes too long to load

- Pre-bake weights into the Docker image
- Use a faster storage backend (NVMe)
- Consider using a network volume on Vast.ai

### Out of Memory (OOM)

- Lower the `WAN2GP_PROFILE` value
- Reduce `num_frames` or resolution
- Use a smaller model (`t2v_1.3B`)

### Generation timeout

- Increase client timeout (default: 600s)
- Reduce `num_inference_steps`
- Use `euler` sampler instead of `unipc`

### Port not accessible

- Ensure port mapping is configured: `-p 8000:8000`
- Check firewall rules on the host
- On Vast.ai, configure "Port Mappings" in the instance settings

## Cost Considerations

### Vast.ai Pricing Model

Vast.ai charges by the hour for GPU rental. Your API will be "always-on" and you pay whether generating or not.

**To minimize costs:**
- Use spot instances for development
- Consider RunPod Serverless or Modal for true pay-per-request
- Implement auto-shutdown when idle

### True Serverless (Pay-per-Second)

For actual serverless (scale to zero), consider:

1. **RunPod Serverless**: Wrap this API as a RunPod handler
2. **Modal**: Deploy using Modal's GPU decorator
3. **Replicate**: Package as a Cog model

These platforms only charge when your model is actively generating.

## Security Notes

- The API has **no authentication** by default
- Add an API key middleware for production use
- Use HTTPS (reverse proxy like nginx or Caddy)
- Consider rate limiting for public deployments

