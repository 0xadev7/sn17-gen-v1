from __future__ import annotations
from typing import Optional
import asyncio
import threading
import torch
from diffusers import FluxPipeline
from PIL import Image

from gen.utils.prompt import tune_prompt


class FluxText2Image:
    def __init__(self, device: torch.device):
        self.device = device
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

        self.pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=dtype,
        )
        self.pipe.to(self.device)

        # (Optional) small perf wins on repeated shapes
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = True

    @torch.inference_mode()
    def generate(
        self, prompt: str, steps: int, guidance: float, res: int, seed: int = 0
    ) -> Image.Image:
        prompt = tune_prompt(prompt)

        gen = torch.Generator(device="cpu").manual_seed(seed)
        out = self.pipe(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            max_sequence_length=256,
            generator=gen,
            height=res,
            width=res,
        )

        return out.images[0]
