"""
LOVBEAR — RunPod Serverless worker (generative After)

Deploy:
  Boot-time curl deploy (see start.sh / RunPod template "lovbear-after"):
  the endpoint pip-installs deps then curls this file from `main` and runs it.
  No Docker build step — just push to this repo's `main` branch.

Input JSON (under "input"):
  image_base64,
  mask_base64,       # optional. White = AI may repaint, black = keep the
                      # original pixel exactly. Sent by the web app so the
                      # edit stays scoped to one facial feature (e.g. nose)
                      # instead of repainting the whole face.
  prompt, negative_prompt, strength, procedure_id, intensity, ...

Output:
  { "image_base64": "<jpeg base64 without data-url prefix>" }

NOTE:
  - With a mask: uses Diffusers SDXL *inpainting* — only the masked region
    is regenerated; everything outside the mask is composited back from the
    original photo pixel-for-pixel (defense in depth against identity drift
    even if the model tries to touch the rest of the face).
  - Without a mask (older client / explicit whole-image request): falls back
    to plain img2img on the full frame, same as before.
  - This is NOT medical advice — non-medical, reference-only simulation.
"""

from __future__ import annotations

import base64
import io
import os
from typing import Any

import runpod


def _b64_to_bytes(value: str) -> bytes:
    raw = value
    if "," in raw and raw.strip().startswith("data:"):
        raw = raw.split(",", 1)[1]
    return base64.b64decode(raw)


def _decode_image(image_base64: str):
    from PIL import Image

    img = Image.open(io.BytesIO(_b64_to_bytes(image_base64))).convert("RGB")
    return img


def _decode_mask(mask_base64: str, size: tuple[int, int]):
    import numpy as np
    from PIL import Image

    m = Image.open(io.BytesIO(_b64_to_bytes(mask_base64)))
    if m.mode == "RGBA":
        # Web client draws a white radial glow on a transparent background —
        # use alpha (or luminance, whichever is stronger) as the mask.
        alpha = np.array(m.split()[-1])
        lum = np.array(m.convert("L"))
        merged = np.maximum(alpha, lum).astype("uint8")
        m = Image.fromarray(merged, "L")
    else:
        m = m.convert("L")
    return m.resize(size, Image.Resampling.BILINEAR)


def _round8(x: int) -> int:
    return max(8, int(round(x / 8.0) * 8))


def _resize_cap(img, max_side: int):
    from PIL import Image

    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    tw, th = _round8(w * scale), _round8(h * scale)
    return img.resize((tw, th), Image.Resampling.LANCZOS)


def _encode_jpeg(img, quality: int = 92) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


_IMG2IMG_PIPE = None
_INPAINT_PIPE = None


def _get_img2img_pipe():
    global _IMG2IMG_PIPE
    if _IMG2IMG_PIPE is not None:
        return _IMG2IMG_PIPE

    import torch
    from diffusers import AutoPipelineForImage2Image

    model_id = os.getenv("MODEL_ID", "stabilityai/stable-diffusion-xl-base-1.0")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    pipe = AutoPipelineForImage2Image.from_pretrained(
        model_id,
        torch_dtype=dtype,
        variant="fp16" if dtype == torch.float16 else None,
        use_safetensors=True,
    )
    if torch.cuda.is_available():
        pipe.enable_attention_slicing()
        pipe = pipe.to("cuda")
    _IMG2IMG_PIPE = pipe
    return _IMG2IMG_PIPE


def _get_inpaint_pipe():
    global _INPAINT_PIPE
    if _INPAINT_PIPE is not None:
        return _INPAINT_PIPE

    import torch
    from diffusers import AutoPipelineForInpainting

    # Dedicated inpainting checkpoint (kept separate from MODEL_ID, which the
    # template also uses for plain img2img) — much better local-edit quality
    # than running a base SDXL checkpoint through the inpaint pipeline.
    model_id = os.getenv(
        "INPAINT_MODEL_ID", "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    )
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    pipe = AutoPipelineForInpainting.from_pretrained(
        model_id,
        torch_dtype=dtype,
        variant="fp16" if dtype == torch.float16 else None,
        use_safetensors=True,
    )
    if torch.cuda.is_available():
        pipe.enable_attention_slicing()
        pipe = pipe.to("cuda")
    _INPAINT_PIPE = pipe
    return _INPAINT_PIPE


def _run_inpaint(init_image, mask_image, prompt, negative, strength, guidance, steps):
    """Masked local edit + high-res composite so pixels outside the mask
    are guaranteed to stay exactly the original (no drift/identity change)."""
    from PIL import Image, ImageFilter
    import numpy as np

    ow, oh = init_image.size
    max_side = int(os.getenv("MAX_SIDE", "768"))

    work_img = _resize_cap(init_image, max_side)
    ww, wh = work_img.size
    work_mask = mask_image.resize((ww, wh), Image.Resampling.BILINEAR)

    pipe = _get_inpaint_pipe()
    result = pipe(
        prompt=prompt,
        negative_prompt=negative,
        image=work_img,
        mask_image=work_mask,
        strength=strength,
        guidance_scale=guidance,
        num_inference_steps=steps,
    )
    gen = result.images[0].resize((ow, oh), Image.Resampling.LANCZOS)

    mask_full = mask_image.resize((ow, oh), Image.Resampling.BILINEAR)
    feather = max(2, min(ow, oh) // 200)
    mask_soft = mask_full.filter(ImageFilter.GaussianBlur(radius=feather))

    orig_arr = np.asarray(init_image).astype(np.float32)
    gen_arr = np.asarray(gen).astype(np.float32)
    m = (np.asarray(mask_soft).astype(np.float32) / 255.0)[:, :, None]

    composited = orig_arr * (1.0 - m) + gen_arr * m
    return Image.fromarray(composited.clip(0, 255).astype("uint8"), "RGB")


def _run_img2img(init_image, prompt, negative, strength, guidance, steps):
    max_side = int(os.getenv("MAX_SIDE", "768"))
    work_img = _resize_cap(init_image, max_side)
    pipe = _get_img2img_pipe()
    result = pipe(
        prompt=prompt,
        negative_prompt=negative,
        image=work_img,
        strength=strength,
        guidance_scale=guidance,
        num_inference_steps=steps,
    )
    return result.images[0].resize(init_image.size)


def handler(event: dict[str, Any]) -> dict[str, Any]:
    job_input = event.get("input") or {}
    image_b64 = job_input.get("image_base64") or job_input.get("imageBase64")
    if not image_b64:
        return {"error": "image_base64 is required"}

    mask_b64 = job_input.get("mask_base64") or job_input.get("maskBase64")

    prompt = job_input.get("prompt") or (
        "same person, photorealistic portrait, subtle natural refinement, keep identity"
    )
    negative = job_input.get("negative_prompt") or (
        "different person, deformed, plastic skin, cartoon, text, watermark"
    )
    strength = float(job_input.get("strength") or 0.35)
    strength = max(0.15, min(0.9, strength))
    steps = int(job_input.get("steps") or os.getenv("STEPS", "20"))
    guidance = float(job_input.get("guidance_scale") or os.getenv("GUIDANCE", "6.5"))

    try:
        init_image = _decode_image(image_b64)
        if mask_b64:
            mask_image = _decode_mask(mask_b64, init_image.size)
            out = _run_inpaint(
                init_image, mask_image, prompt, negative, strength, guidance, steps
            )
        else:
            out = _run_img2img(init_image, prompt, negative, strength, guidance, steps)

        return {
            "image_base64": _encode_jpeg(out),
            "procedure_id": job_input.get("procedure_id"),
            "mode": "inpaint" if mask_b64 else "img2img",
            "purpose": "non_medical_image_simulation",
        }
    except Exception as exc:  # noqa: BLE001 — surface to RunPod status
        return {
            "error": f"generation_failed: {type(exc).__name__}: {exc}",
        }


runpod.serverless.start({"handler": handler})
