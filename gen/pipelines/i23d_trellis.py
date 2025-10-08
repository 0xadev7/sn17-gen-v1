from __future__ import annotations
import io, torch, os
from typing import List, Optional, Tuple
from PIL import Image

from gen.lib.trellis.pipelines import TrellisImageTo3DPipeline


class TrellisImageTo3D:
    def __init__(
        self,
        device: torch.device,
        steps_struct: int,
        steps_slat: int,
        cfg_struct: float,
        cfg_slat: float,
        max_gs: int,
        target_mb: int,
    ):
        self.pipe = TrellisImageTo3DPipeline.from_pretrained(
            "microsoft/TRELLIS-image-large"
        )
        self.pipe.cuda()

        self.steps_struct = steps_struct
        self.steps_slat = steps_slat
        self.cfg_struct = cfg_struct
        self.cfg_slat = cfg_slat
        self.max_gs = max_gs
        self.target_mb = target_mb

    @torch.inference_mode()
    async def infer_to_ply(
        self,
        image: Image.Image,
        struct_steps=None,
        slat_steps=None,
        cfg_struct=None,
        cfg_slat=None,
        seed=None,
    ) -> bytes:
        outputs = self.pipe.run(
            image,
            seed=seed if seed is not None else 1,
            sparse_structure_sampler_params={
                "steps": (
                    struct_steps if struct_steps is not None else self.steps_struct
                ),
                "cfg_strength": (
                    cfg_struct if cfg_struct is not None else self.cfg_struct
                ),
            },
            slat_sampler_params={
                "steps": slat_steps if slat_steps is not None else self.steps_slat,
                "cfg_strength": cfg_slat if cfg_slat is not None else self.cfg_slat,
            },
        )
        gs = outputs["gaussian"][0]

        # Export to PLY in-memory
        buf = io.BytesIO()
        gs.save_ply(buf)
        return buf.getvalue()
