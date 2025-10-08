from __future__ import annotations
import torch
from PIL import Image
import numpy as np
from transformers import AutoModelForImageSegmentation, AutoImageProcessor


class BiRefNetMatte:
    def __init__(self, device: torch.device):
        self.processor = AutoImageProcessor.from_pretrained(
            "ZhengPeng7/BiRefNet", trust_remote_code=True
        )
        self.model = (
            AutoModelForImageSegmentation.from_pretrained(
                "ZhengPeng7/BiRefNet", trust_remote_code=True
            )
            .to(device)
            .eval()
        )
        self.device = device

    @torch.inference_mode()
    async def remove_bg(self, image: Image.Image) -> Image.Image:
        # outputs single-channel matting mask
        inputs = self.processor(images=image, return_tensors="pt").to(
            self.device, dtype=torch.float16
        )
        logits = self.model(**inputs).logits  # (1,1,H,W) per the trust_remote_code
        mask = torch.sigmoid(logits.float())[0, 0].cpu().numpy()
        mask = np.clip(mask, 0, 1)

        im = np.array(image.convert("RGBA"), dtype=np.float32) / 255.0
        im[..., 3] = mask  # put matte in alpha
        im = (im * 255).astype(np.uint8)
        return Image.fromarray(im, mode="RGBA")
