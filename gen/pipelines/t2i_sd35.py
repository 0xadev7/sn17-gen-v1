from __future__ import annotations
from typing import Optional
import asyncio
import threading
import torch
from diffusers import StableDiffusion3Pipeline
from PIL import Image

from gen.utils.prompt import tune_prompt


class SD35Text2Image:
    def __init__(self, device: torch.device):
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        self.device = device

        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3.5-large-turbo", torch_dtype=dtype
        )

        self.pipe.to(self.device)

    @torch.inference_mode()
    def generate(
        self, prompt: str, steps: int, guidance: float, res: int, seed: int = 0
    ) -> Image.Image:
        prompt = tune_prompt(prompt)

        if seed is not None:
            torch.manual_seed(seed)

        width = (res // 16) * 16
        height = (res // 16) * 16

        out = self.pipe(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=0.0,
            width=width,
            height=height,
        )

        return out.images[0]
