from typing import List, Tuple
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation


class BiRefNetRemover:
    def __init__(self, device: torch.device):
        self.device = device
        self.model = (
            AutoModelForImageSegmentation.from_pretrained(
                "ZhengPeng7/BiRefNet", trust_remote_code=True
            )
            .to(self.device)
            .eval()
        )

        if self.device.type == "cuda":
            self.model.half()
            torch.set_float32_matmul_precision("high")

        self.image_size = (1024, 1024)
        self.tfm = transforms.Compose(
            [
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    @torch.inference_mode()
    def remove(self, img: Image.Image) -> Tuple[Image.Image, np.ndarray]:
        rgb = img.convert("RGB")
        x = self.tfm(rgb).unsqueeze(0).to(self.device)
        if self.device.type == "cuda":
            x = x.half()

        pred = self.model(x)[-1].sigmoid().float().cpu()[0, 0]  # HxW
        mask_pil = transforms.functional.resize(transforms.ToPILImage()(pred), rgb.size)

        out = rgb.copy()
        out.putalpha(mask_pil)

        alpha = np.array(mask_pil, dtype=np.float32) / 255.0
        return out, alpha

    def remove_batch(
        self,
        images: List[Image.Image],
        microbatch_size: int = 3,
    ) -> List[Image.Image]:
        """
        Batched(ish) BG removal. If your implementation supports real batch, replace the loop
        with a vectorized forward. Otherwise we loop in micro-batches to avoid VRAM spikes.
        Returns a list of RGBA PIL images of same length as input.
        """
        results: List[Image.Image] = []
        n = len(images)
        mb = max(1, microbatch_size)

        with torch.inference_mode():
            i = 0
            while i < n:
                chunk = images[i : i + mb]
                # If you have a true batched API (e.g., self.remove_batch_internal), use it here.
                # Otherwise, loop but keep it under inference guard.
                for img in chunk:
                    # Ensure format (some T2I might output RGB)
                    if img.mode != "RGBA":
                        img = img.convert("RGBA")
                    fg, _ = self.remove(img)  # expected to be sync
                    if fg.mode != "RGBA":
                        fg = fg.convert("RGBA")
                    results.append(fg)
                i += mb
        return results
