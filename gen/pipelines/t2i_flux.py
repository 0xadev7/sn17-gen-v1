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
        suffix = ", high quality single 3d game object centered"
        return f"{prompt.strip().replace('.', ' ')} {suffix}"

    @torch.inference_mode()
    async def generate(
        self, prompt: str, steps: int, guidance: float, res: int, seed=0
    ) -> Image.Image:
        prompt = self.tune_prompt(prompt)
        out = self.pipe(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            max_sequence_length=256,
            generator=torch.Generator("cpu").manual_seed(seed),
            height=res,
            width=res,
        )
        return out.images[0]
