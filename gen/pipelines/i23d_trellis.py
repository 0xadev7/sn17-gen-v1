from __future__ import annotations
import io, torch, os
from typing import List, Tuple
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
    async def infer_to_ply(self, image: Image.Image) -> bytes:
        outputs = self.pipe.run(
            image,
            seed=1,
            sparse_structure_sampler_params={
                "steps": self.steps_struct,
                "cfg_strength": self.cfg_struct,
            },
            slat_sampler_params={
                "steps": self.steps_slat,
                "cfg_strength": self.cfg_slat,
            },
        )
        gs = outputs["gaussian"][0]  # Trellis returns first result

        # Control size: prune/densify to fit max points and approx target MB
        gs.prune_to_max_points(self.max_gs)

        # Export to PLY in-memory
        buf = io.BytesIO()
        gs.save_ply(buf)
        return buf.getvalue()
