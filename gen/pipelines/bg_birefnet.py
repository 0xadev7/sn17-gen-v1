from typing import Tuple
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

from gen.utils.vram import vram_guard


class BiRefNetRemover:
    def __init__(self, device: torch.device):
        self.device = device
        self.dtype = torch.float16 if device.type == "cuda" else torch.float32
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

        with vram_guard():
            x = self.tfm(rgb).unsqueeze(0).to(self.device, non_blocking=True)
            if self.device.type == "cuda":
                ctx = torch.autocast(device_type="cuda", dtype=self.dtype)
            else:

                class _Noop:
                    def __enter__(self):
                        pass

                    def __exit__(self, *a):
                        pass

                ctx = _Noop()

            try:
                with ctx:
                    pred = self.model(x)[-1].sigmoid()  # (B,1,H,W) on device/float16
                # Move once to CPU as float32; release device tensors immediately.
                pred_cpu = pred.float().cpu()[0, 0]  # HxW, torch.float32 on CPU
            finally:
                # Drop GPU tensors promptly
                del x
                del pred

            mask_pil = transforms.functional.resize(
                transforms.ToPILImage()(pred_cpu), rgb.size
            )

            out = rgb.copy()
            out.putalpha(mask_pil)

            alpha = np.array(mask_pil, dtype=np.float32) / 255.0
            return out, alpha
