from __future__ import annotations
from typing import Optional
import asyncio
import threading
import torch
from diffusers import StableDiffusion3Pipeline
from PIL import Image


class SD35Text2Image:
    def __init__(self, device: torch.device):
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3.5-large-turbo", torch_dtype=dtype
        )

        self.pipe.to(device)
        self.device = device

        # Protect pipeline calls; Diffusers/CUDA aren’t safely concurrent per instance.
        self._lock = threading.Lock()

        # (Optional) small perf wins on repeated shapes
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = True

        # Reduce peak memory a bit
        try:
            self.pipe.enable_attention_slicing()
        except Exception:
            pass

    def tune_prompt(self, base_prompt: str) -> str:
        return (
            f"{base_prompt}, high quality 3D object photo, "
            "on a contrasting neutral background, "
            "studio lighting, sharp focus, centered composition"
        )

    @torch.inference_mode()
    def generate_sync(
        self, prompt: str, steps: int, guidance: float, res: int, seed: int = 0
    ) -> Image.Image:
        """Blocking generation (runs on CPU thread; safe to call inside to_thread)."""
        prompt = self.tune_prompt(prompt)

        # Ensure ops happen on the right CUDA device even if other threads are busy
        with self._lock:
            if self.device.type == "cuda":
                torch.cuda.set_device(self.device)

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

    async def generate(
        self, prompt: str, steps: int, guidance: float, res: int, seed: int = 0
    ) -> Image.Image:
        """Non-blocking wrapper suitable for asyncio pipelines."""
        return await asyncio.to_thread(
            self.generate_sync, prompt, steps, guidance, res, seed
        )
