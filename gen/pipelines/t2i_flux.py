from __future__ import annotations
import torch
from diffusers import FluxPipeline
from transformers import T5EncoderModel, AutoTokenizer
from PIL import Image


class FluxText2Image:
    def __init__(self, device: torch.device):
        tokenizer = AutoTokenizer.from_pretrained("t5-base")

        text_encoder = T5EncoderModel.from_pretrained(
            "t5-base",
        )

        self.pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=torch.bfloat16,
            device_map="balanced",
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )
        self.pipe.to(device)
        self.device = device

    @torch.inference_mode()
    async def generate(
        self, prompt: str, steps: int, guidance: float, res: int
    ) -> Image.Image:
        out = self.pipe(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            height=res,
            width=res,
        )
        return out.images[0]
