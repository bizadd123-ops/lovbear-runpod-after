"""
LOVBEAR — RunPod Serverless worker (generative After)

Deploy:
  1. Create Serverless endpoint on RunPod (GPU, e.g. RTX 4090 / A40)
  2. Use this folder as the worker (or bake into a Docker image with Diffusers)
  3. Set env on endpoint if needed: HF_TOKEN, MODEL_ID

Input JSON (under "input"):
  image_base64, prompt, negative_prompt, strength, procedure_id, intensity, ...

Output:
  { "image_base64": "<jpeg base64 without data-url prefix>" }

NOTE:
  - Default path uses Diffusers img2img when torch/diffusers are available.
  - If models are not installed, returns a clear error so the app can fall back.
  - This is NOT medical advice — simulation only.
"""

from __future__ import annotations

import base64
import io
import os
from typing import Any

import runpod


def _decode_image(image_base64: str):
    from PIL import Image

    raw = image_base64
    if "," in raw and raw.strip().startswith("data:"):
        raw = raw.split(",", 1)[1]
    data = base64.b64decode(raw)
    img = Image.open(io.BytesIO(data)).convert("RGB")
    # Cap size for speed/cost
    max_side = int(os.getenv("MAX_SIDE", "768"))
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    return img


def _encode_jpeg(img, quality: int = 92) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


_PIPE = None


def _get_pipe():
    """Lazy-load img2img pipeline once per worker."""
    global _PIPE
    if _PIPE is not None:
        return _PIPE

    import torch
    from diffusers import AutoPipelineForImage2Image

    model_id = os.getenv(
        "MODEL_ID",
        "stabilityai/stable-diffusion-xl-base-1.0",
    )
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    pipe = AutoPipelineForImage2Image.from_pretrained(
        model_id,
        torch_dtype=dtype,
        variant="fp16" if dtype == torch.float16 else None,
        use_safetensors=True,
    )
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    _PIPE = pipe
    return _PIPE


def handler(event: dict[str, Any]) -> dict[str, Any]:
    job_input = event.get("input") or {}
    image_b64 = job_input.get("image_base64") or job_input.get("imageBase64")
    if not image_b64:
        return {"error": "image_base64 is required"}

    prompt = job_input.get("prompt") or (
        "same person, photorealistic portrait, subtle natural refinement, keep identity"
    )
    negative = job_input.get("negative_prompt") or (
        "different person, deformed, plastic skin, cartoon, text, watermark"
    )
    strength = float(job_input.get("strength") or 0.35)
    strength = max(0.15, min(0.6, strength))
    steps = int(job_input.get("steps") or os.getenv("STEPS", "28"))

    try:
        init_image = _decode_image(image_b64)
        pipe = _get_pipe()
        result = pipe(
            prompt=prompt,
            negative_prompt=negative,
            image=init_image,
            strength=strength,
            guidance_scale=float(os.getenv("GUIDANCE", "5.5")),
            num_inference_steps=steps,
        )
        out = result.images[0]
        return {
            "image_base64": _encode_jpeg(out),
            "procedure_id": job_input.get("procedure_id"),
            "purpose": "non_medical_image_simulation",
        }
    except Exception as exc:  # noqa: BLE001 — surface to RunPod status
        return {
            "error": f"generation_failed: {type(exc).__name__}: {exc}",
        }


runpod.serverless.start({"handler": handler})
