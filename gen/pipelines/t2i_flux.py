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

    def tune_prompt(self, base_prompt: str) -> str:
        p = base_prompt.lower()
        if any(c in p for c in ["red", "pink", "orange", "rose", "coral"]):
            bg = "light blue background"
        elif any(c in p for c in ["yellow", "beige", "cream", "gold"]):
            bg = "soft gray background"
        elif any(c in p for c in ["green", "emerald", "mint", "lime"]):
            bg = "light peach background"
        elif any(c in p for c in ["blue", "teal", "navy", "cyan"]):
            bg = "warm cream background"
        elif any(c in p for c in ["purple", "violet", "lavender"]):
            bg = "light yellow background"
        elif any(c in p for c in ["black", "dark", "charcoal"]):
            bg = "light gray background"
        elif any(c in p for c in ["white", "ivory", "pale", "light"]):
            bg = "dark slate background"
        elif any(c in p for c in ["metal", "silver", "chrome", "platinum"]):
            bg = "matte neutral gray background"
        else:
            bg = "neutral gray background"

        return f"{base_prompt}, {bg}, high quality single object centered"

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
