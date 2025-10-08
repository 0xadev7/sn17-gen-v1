from __future__ import annotations
from typing import Tuple
from loguru import logger
import torch

from gen.settings import Config
from gen.pipelines.t2i_flux import FluxText2Image
from gen.pipelines.bg_birefnet import BiRefNetRemover
from gen.pipelines.i23d_trellis import TrellisImageTo3D
from gen.validators.external_validator import ExternalValidator


class MinerState:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        torch.cuda.set_device(cfg.gpu_id)
        self.device = torch.device(f"cuda:{cfg.gpu_id}")

        # Pipelines
        self.t2i = FluxText2Image(self.device)
        self.bg_remover = BiRefNetRemover(self.device)
        self.trellis_img = TrellisImageTo3D(
            self.device,
            cfg.trellis_struct_steps,
            cfg.trellis_slat_steps,
            cfg.trellis_cfg_struct,
            cfg.trellis_cfg_slat,
            cfg.trellis_max_gaussians,
            cfg.trellis_target_mb,
        )

        # Validators
        self.validator = ExternalValidator(
            cfg.validator_url_txt, cfg.validator_url_img, cfg.vld_threshold
        )

        logger.info("Models loaded and warmed-up.")

    async def text_to_ply(self, prompt: str) -> Tuple[bytes, float]:
        # 1) Text→image (fast)
        image = await self.t2i.generate(
            prompt,
            steps=self.cfg.t2i_steps,
            guidance=self.cfg.t2i_guidance,
            res=self.cfg.t2i_res,
        )

        # 2) Background removal
        fg, _ = self.bg_remover.remove(image)

        # 4) TRELLIS image-to-3D
        ply_bytes = await self.trellis_img.infer_to_ply(fg)

        # 5) Validation
        score, passed, _ = await self.validator.validate_text(prompt, ply_bytes)
        logger.info(f"External validator (text): score={score}, passed={passed}")

        return ply_bytes if passed else b"", score

    async def image_to_ply(self, pil_image) -> Tuple[bytes, float]:
        # 1) BG removal
        fg, _ = self.bg_remover.remove(pil_image)

        # 2) TRELLIS image-to-3D
        ply_bytes = await self.trellis_img.infer_to_ply(fg)

        # 3) Validation
        score, passed, _ = await self.validator.validate_image(pil_image, ply_bytes)
        logger.info(f"External validator (image): score={score}, passed={passed}")

        return ply_bytes if passed else b"", score
