from typing import Tuple
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation


class BiRefNetRemover:
    def __init__(
        self, model_id: str = "ZhengPeng7/BiRefNet", device: str | None = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = (
            AutoModelForImageSegmentation.from_pretrained(
                model_id, trust_remote_code=True
            )
            .to(self.device)
            .eval()
        )
        if self.device == "cuda":
            self.model.half()
            torch.set_float32_matmul_precision("high")

        # Use 1024 for general; dynamic tolerates any size, but we keep a stable load.
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
        if self.device == "cuda":
            x = x.half()

        pred = self.model(x)[-1].sigmoid().float().cpu()[0, 0]  # HxW
        mask_pil = transforms.functional.resize(transforms.ToPILImage()(pred), rgb.size)
        # 0–255 alpha
        out = rgb.copy()
        out.putalpha(mask_pil)

        alpha = np.array(mask_pil, dtype=np.float32) / 255.0  # HxW in [0,1]
        return out, alpha
