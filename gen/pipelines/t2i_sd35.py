from __future__ import annotations
from typing import Optional
import torch
from diffusers import StableDiffusion3Pipeline
from PIL import Image
from gen.utils.prompt import tune_prompt
from gen.utils.vram import vram_guard


class SD35Text2Image:
    def __init__(self, device: torch.device):
        self.device = device
        self.dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3.5-large-turbo", torch_dtype=self.dtype
        ).to(self.device)

        # Safer memory footprint on long-running loops
        if self.device.type == "cuda":
            # These are no-ops if unsupported
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
        self, prompt: str, steps: int, guidance: float, res: int, seed: int = 0
    ) -> Image.Image:
        prompt = tune_prompt(prompt)

        width = (res // 16) * 16
        height = (res // 16) * 16

        generator = torch.Generator(
            device=self.device.type if self.device.type == "cuda" else "cpu"
        )
        if seed is not None:
            generator.manual_seed(int(seed))

        with vram_guard():
            if self.device.type == "cuda":
                autocast_ctx = torch.amp.autocast(dtype=self.dtype)
            else:

                class _Noop:
                    def __enter__(self):
                        pass

                    def __exit__(self, *a):
                        pass

                autocast_ctx = _Noop()

            with autocast_ctx:
                out = self.pipe(
                    prompt=prompt,
                    num_inference_steps=int(steps),
                    guidance_scale=float(guidance or 0.0),
                    width=int(width),
                    height=int(height),
                    generator=generator,
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
