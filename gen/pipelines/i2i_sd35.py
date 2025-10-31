from __future__ import annotations
from typing import Optional
from contextlib import nullcontext
import torch
from diffusers import StableDiffusion3Img2ImgPipeline
from PIL import Image
from gen.utils.prompt import refactor_prompt
from gen.utils.vram import vram_guard


class SD35Image2Image:
    def __init__(self, device: torch.device):
        self.device = device
        self.dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

        self.pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-3.5-large-turbo",
            torch_dtype=self.dtype,
        ).to(self.device)

        if self.device.type == "cuda":
            # These calls are safe no-ops if unsupported by the pipeline
            try:
                self.pipe.enable_vae_tiling()
            except Exception:
                pass
            try:
                self.pipe.enable_attention_slicing("max")
            except Exception:
                pass

        self.pipe.set_progress_bar_config(disable=True)

    @torch.inference_mode()
    def generate(
        self,
        source: Image.Image,
        steps: int,
        guidance: float,
        res: int,
        strength: float = 0.3,
        seed: int = 0,
    ) -> Image.Image:
        src_rgb = source.convert("RGB")
        prompt_pos = (
            "studio product photo, single centered object, 3/4 view, neutral gray gradient backdrop, "
            "neutral three-point lighting with soft rim light, sharp focus, physically-plausible materials, "
            "accurate proportions, clean silhouette, high microdetail, true color, "
            "50mm lens equivalent, even neutral studio lighting"
        )
        prompt_neg = (
            "no humans, no hands, no text, no logos, no labels, no watermark, "
            "no background scene, no environment, no reflections of room, no depth of field blur, "
            "no motion blur, no duplicate objects, no overexposure, no underexposure, "
            "no extreme contrast, no heavy color grading, no fog, no smoke"
        )

        # Keep sizes divisible by 16 for safety with SD3.5
        width = (int(res) // 16) * 16
        height = (int(res) // 16) * 16

        gen = torch.Generator(device="cuda" if self.device.type == "cuda" else "cpu")
        if seed is not None:
            gen.manual_seed(int(seed))

        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=self.dtype)
            if self.device.type == "cuda"
            else nullcontext()
        )

        with vram_guard():
            out = None
            img = None
            try:
                with autocast_ctx:
                    out = self.pipe(
                        prompt=prompt_pos,
                        negative_prompt=prompt_neg,
                        image=src_rgb,
                        strength=float(strength),
                        guidance_scale=float(guidance),
                        num_inference_steps=int(steps),
                        width=width,
                        height=height,
                        generator=gen,
                    )
                img = out.images[0]
                result = img.copy()  # decouple from pipeline internals
                return result
            finally:
                # Only touch objects that exist
                try:
                    if img is not None:
                        img.close()
                except Exception:
                    pass
                try:
                    if out is not None:
                        del out
                except Exception:
                    pass

                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
