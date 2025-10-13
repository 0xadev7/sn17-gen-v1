from __future__ import annotations
from typing import List, Optional

import asyncio
from functools import partial
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
        return (
            f"{base_prompt}, high quality 3D object photo, "
            "on a contrasting neutral background, "
            "studio lighting, sharp focus, centered composition, "
        )

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

    async def generate_batch(
        self,
        prompt: str,
        seeds: List[int],
        steps: int,
        guidance: float,
        res: int,
    ) -> List[Image.Image]:
        """
        Try to generate N images in a single forward pass. If the wrapped pipeline
        doesn't support batch, fall back to a tight sequential loop.
        Returns a list of PIL Images (RGBA or RGB), length == len(seeds).
        """
        height = res
        width = res
        n = len(seeds)

        # If you know your model prefers fp16 on A100, switch dtype to torch.float16.
        # bfloat16 is usually safe and accurate on A100.
        amp_dtype = torch.bfloat16

        # Path A: true batch via diffusers-like API
        if hasattr(self, "pipe") and self.pipe is not None:
            try:
                gens = [
                    torch.Generator(device=self.device).manual_seed(s) for s in seeds
                ]

                def _run_pipe():
                    with torch.inference_mode(), torch.autocast(
                        "cuda", dtype=amp_dtype
                    ):
                        out = self.pipe(
                            prompt=[prompt] * n,
                            num_inference_steps=steps,
                            guidance_scale=guidance,
                            height=height,
                            width=width,
                            generator=gens,  # list of per-sample RNGs
                            num_images_per_prompt=1,  # each prompt → one image
                            output_type="pil",
                        )
                    # diffusers returns `out.images` as a list[PIL.Image]
                    return list(out.images)

                # diffusers call is sync; run in threadpool so our API stays async
                loop = asyncio.get_running_loop()
                images = await loop.run_in_executor(None, _run_pipe)
                return images
            except Exception as e:
                # fall back to sequential
                print(
                    f"[FluxText2Image.generate_batch] Batched path failed, falling back: {e}"
                )

        # Path B: tight sequential fallback (still uses AMP, avoids device churn)
        out_images: List[Image.Image] = []
        # If your class already has an async `generate`, reuse it (fastest).
        if hasattr(self, "generate") and callable(getattr(self, "generate")):
            tasks = [
                self.generate(prompt, steps=steps, guidance=guidance, res=res, seed=s)
                for s in seeds
            ]
            # limit concurrency to 1 on single GPU to avoid time-slicing
            for t in tasks:
                img = await t
                out_images.append(img)
            return out_images

        # If you only have a sync callable (self.pipe or self.forward)
        def _run_single(seed: int):
            g = torch.Generator(device=self.device).manual_seed(seed)
            with torch.inference_mode(), torch.autocast("cuda", dtype=amp_dtype):
                out = self.pipe(
                    prompt=prompt,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    height=height,
                    width=width,
                    generator=g,
                    num_images_per_prompt=1,
                    output_type="pil",
                )
            return out.images[0] if hasattr(out, "images") else out[0]

        loop = asyncio.get_running_loop()
        for s in seeds:
            img = await loop.run_in_executor(None, partial(_run_single, s))
            out_images.append(img)
        return out_images
