#!/usr/bin/env python3
"""
Wan2GP API Client Example

This script demonstrates how to interact with the Wan2GP API server
for video generation tasks.

Usage:
    # Text-to-Video
    python api_client.py --prompt "A cat walking in a garden" --output cat_video.mp4

    # Image-to-Video
    python api_client.py --prompt "A person dancing" --image input.jpg --output dance.mp4

    # Custom server
    python api_client.py --url http://YOUR_VAST_IP:8000 --prompt "Ocean waves"
"""

import argparse
import base64
import json
import time
from pathlib import Path
from typing import Optional

import requests


class Wan2GPClient:
    """Client for interacting with the Wan2GP API server"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def health_check(self) -> dict:
        """Check server health and model status"""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    def text_to_video(
        self,
        prompt: str,
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
    ) -> dict:
        """
        Generate a video from a text prompt.
        
        Returns:
            dict with status, job_id, video_url, generation_time_seconds
        """
        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "flow_shift": flow_shift,
            "seed": seed,
            "sample_solver": sample_solver,
            "fps": fps,
        }
        
        response = self.session.post(
            f"{self.base_url}/generate/t2v",
            json=payload,
            timeout=600,  # 10 minute timeout for generation
        )
        response.raise_for_status()
        return response.json()

    def image_to_video(
        self,
        prompt: str,
        image_path: str,
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
    ) -> dict:
        """
        Generate a video from an image and text prompt.
        
        Args:
            image_path: Path to the input image (PNG/JPEG)
            
        Returns:
            dict with status, job_id, video_url, generation_time_seconds
        """
        # Read and encode image
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        
        payload = {
            "prompt": prompt,
            "image_base64": image_base64,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "flow_shift": flow_shift,
            "seed": seed,
            "sample_solver": sample_solver,
            "fps": fps,
        }
        
        response = self.session.post(
            f"{self.base_url}/generate/i2v",
            json=payload,
            timeout=600,
        )
        response.raise_for_status()
        return response.json()

    def download_video(self, video_url: str, output_path: str) -> str:
        """
        Download a generated video to a local file.
        
        Args:
            video_url: The video URL from the generation response
            output_path: Local path to save the video
            
        Returns:
            The absolute path to the saved video
        """
        full_url = f"{self.base_url}{video_url}"
        response = self.session.get(full_url, stream=True)
        response.raise_for_status()
        
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return str(Path(output_path).absolute())

    def delete_video(self, job_id: str) -> dict:
        """Delete a generated video from the server"""
        response = self.session.delete(f"{self.base_url}/videos/{job_id}")
        response.raise_for_status()
        return response.json()

    def reload_model(self, model_type: str = "t2v", profile: int = 5) -> dict:
        """Reload the model on the server (for switching model types)"""
        response = self.session.post(
            f"{self.base_url}/reload",
            params={"model_type": model_type, "profile": profile},
        )
        response.raise_for_status()
        return response.json()


def main():
    parser = argparse.ArgumentParser(description="Wan2GP API Client")
    parser.add_argument("--url", type=str, default="http://localhost:8000",
                       help="API server URL")
    parser.add_argument("--prompt", type=str, required=True,
                       help="Text prompt for video generation")
    parser.add_argument("--negative-prompt", type=str, default="",
                       help="Negative prompt")
    parser.add_argument("--image", type=str, default=None,
                       help="Input image for image-to-video")
    parser.add_argument("--output", type=str, default="output.mp4",
                       help="Output video path")
    parser.add_argument("--width", type=int, default=832,
                       help="Video width")
    parser.add_argument("--height", type=int, default=480,
                       help="Video height")
    parser.add_argument("--num-frames", type=int, default=81,
                       help="Number of frames")
    parser.add_argument("--steps", type=int, default=30,
                       help="Number of inference steps")
    parser.add_argument("--guidance", type=float, default=5.0,
                       help="Guidance scale")
    parser.add_argument("--seed", type=int, default=-1,
                       help="Random seed (-1 for random)")
    parser.add_argument("--health-only", action="store_true",
                       help="Only check server health")
    
    args = parser.parse_args()
    
    client = Wan2GPClient(args.url)
    
    # Health check
    print(f"🔍 Checking server health at {args.url}...")
    try:
        health = client.health_check()
        print(f"   Status: {health['status']}")
        print(f"   Model: {health.get('model_type', 'N/A')}")
        print(f"   GPU: {health.get('gpu_name', 'N/A')}")
        if health.get('gpu_memory_total_mb'):
            print(f"   VRAM: {health['gpu_memory_used_mb']}MB / {health['gpu_memory_total_mb']}MB")
        
        if not health['model_loaded']:
            print("⚠️  Model not loaded on server!")
            return 1
            
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to server at {args.url}")
        return 1
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return 1
    
    if args.health_only:
        return 0
    
    print()
    
    # Generate video
    try:
        if args.image:
            print(f"🎬 Generating image-to-video...")
            print(f"   Image: {args.image}")
            result = client.image_to_video(
                prompt=args.prompt,
                image_path=args.image,
                negative_prompt=args.negative_prompt,
                width=args.width,
                height=args.height,
                num_frames=args.num_frames,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                seed=args.seed,
            )
        else:
            print(f"🎬 Generating text-to-video...")
            result = client.text_to_video(
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                width=args.width,
                height=args.height,
                num_frames=args.num_frames,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                seed=args.seed,
            )
        
        print(f"   Prompt: {args.prompt[:50]}...")
        print(f"   Resolution: {args.width}x{args.height}, Frames: {args.num_frames}")
        
        if result['status'] == 'success':
            print(f"✅ Generation complete in {result['generation_time_seconds']}s")
            
            # Download video
            print(f"📥 Downloading video...")
            local_path = client.download_video(result['video_url'], args.output)
            print(f"✅ Video saved to: {local_path}")
            
            # Optionally clean up server storage
            # client.delete_video(result['job_id'])
            
        else:
            print(f"❌ Generation failed: {result.get('message', 'Unknown error')}")
            return 1
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())


