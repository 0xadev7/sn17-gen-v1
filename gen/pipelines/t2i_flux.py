from __future__ import annotations
from typing import Optional
import torch
from contextlib import nullcontext
from diffusers import FluxPipeline
from PIL import Image

from gen.utils.prompt import tune_prompt
from gen.utils.vram import vram_guard


class FluxText2Image:
    def __init__(self, device: torch.device):
        self.device = device
        self.dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

        self.pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=self.dtype,
        ).to(self.device)

        # Optional perf knob
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = True

        # Optional: reduce VRAM spikes on big resolutions
        self.pipe.enable_attention_slicing()

    @torch.inference_mode()
    def generate(
        self, prompt: str, steps: int, guidance: float, res: int, seed: int = 0
    ) -> Image.Image:
        prompt = tune_prompt(prompt)

        with vram_guard():
            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=self.dtype)
                if self.device.type == "cuda"
                else nullcontext()
            )

            out = None
            img = None
            result = None

            try:
                with autocast_ctx:
                    generator = torch.Generator(
                        device="cuda" if self.device.type == "cuda" else "cpu"
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

                # Safely extract & decouple from diffusers internals
                img = out.images[0]
                result = img.copy()
                return result

            finally:
                # Defensive cleanup only if assigned
                try:
                    if img is not None:
                        img.close()
                except Exception:
                    pass
                try:
                    del out
                except Exception:
                    pass

                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
