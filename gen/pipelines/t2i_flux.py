from __future__ import annotations
from typing import Optional
import torch
from diffusers import FluxPipeline
from PIL import Image


class FluxText2Image:
    def __init__(self, device: torch.device):
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        self.pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=dtype,
        )
        self.pipe.to(device)
        self.device = device

    def tune_prompt(self, prompt: str) -> str:
        suffix = ", neutral background, single centered object"
        return f"{prompt.strip()} {suffix}"

    @torch.inference_mode()
    async def generate(
        self, prompt: str, steps: int, guidance: float, res: int
    ) -> Image.Image:
        prompt = self.tune_prompt(prompt)
        out = self.pipe(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            height=res,
            width=res,
        )
        return out.images[0]
