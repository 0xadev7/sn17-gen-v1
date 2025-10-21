from __future__ import annotations
from typing import Optional
import asyncio
import threading
import torch
from diffusers import FluxPipeline
from PIL import Image

from gen.utils.prompt import tune_prompt
from gen.utils.vram import vram_guard


class FluxText2Image:
    def __init__(self, device: torch.device):
        self.device = device
        dtype = torch.float16 if device.type == "cuda" else torch.float32

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

        with vram_guard():
            if self.device.type == "cuda":
                autocast_ctx = torch.autocast(device_type="cuda", dtype=self.dtype)
            else:

                class _Noop:
                    def __enter__(self):
                        pass

                    def __exit__(self, *a):
                        pass

                autocast_ctx = _Noop()

            with autocast_ctx:
                generator = torch.Generator(
                    device=self.device.type if self.device.type == "cuda" else "cpu"
                ).manual_seed(seed)
                out = self.pipe(
                    prompt=prompt,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    max_sequence_length=256,
                    generator=generator,
                    height=res,
                    width=res,
                )

            try:
                img = out.images[0]
                # Return a decoupled copy so diffusers internals can be freed
                result = img.copy()
            finally:
                # Be explicit: large objects often hold GPU refs
                del out
                img.close()

            return result
