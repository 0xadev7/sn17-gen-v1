from typing import Tuple
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
from contextlib import nullcontext

from gen.utils.vram import vram_guard


class BiRefNetRemover:
    def __init__(self, device: torch.device):
        self.device = device
        # Use bf16 for autocast on CUDA, keep fp32 on CPU.
        self.amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

        self.model = (
            AutoModelForImageSegmentation.from_pretrained(
                "ZhengPeng7/BiRefNet", trust_remote_code=True
            )
            .to(self.device, dtype=torch.float32)
            .eval()
        )

        if self.device.type == "cuda":
            # Optional perf knob; no impact on dtype mismatch
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

        x = None
        pred = None

        with vram_guard():
            try:
                # Keep input as fp32; autocast will downcast as needed
                x = (
                    self.tfm(rgb)
                    .unsqueeze(0)
                    .to(self.device, dtype=torch.float32, non_blocking=True)
                )

                amp_ctx = (
                    torch.cuda.amp.autocast(dtype=self.amp_dtype)
                    if self.device.type == "cuda"
                    else nullcontext()
                )

                with amp_ctx:
                    out = self.model(x)
                    # Robustly get logits
                    if isinstance(out, (tuple, list)):
                        logits = out[-1]
                    elif isinstance(out, dict):
                        logits = (
                            out.get("logits")
                            or out.get("out")
                            or next(iter(out.values()))
                        )
                    else:
                        logits = out

                    pred = logits.sigmoid()

                pred_cpu = pred.detach().float().cpu()[0, 0]

            finally:
                # Drop device tensors promptly if they exist
                if x is not None:
                    del x
                if pred is not None and pred.is_cuda:
                    del pred

                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

        # Build alpha mask & RGBA output
        mask_pil = transforms.functional.to_pil_image(pred_cpu.clamp_(0, 1))
        if mask_pil.size != rgb.size:
            mask_pil = transforms.functional.resize(mask_pil, rgb.size)

        out = rgb.copy()
        out.putalpha(mask_pil)

        alpha = np.array(mask_pil, dtype=np.float32) / 255.0
        return out, alpha
