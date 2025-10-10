from __future__ import annotations
import io, torch
from typing import Optional
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
    ):
        self.device = device
        self.pipe = TrellisImageTo3DPipeline.from_pretrained(
            "microsoft/TRELLIS-image-large"
        )
        self.pipe.to(self.device)

        self.steps_struct = steps_struct
        self.steps_slat = steps_slat
        self.cfg_struct = cfg_struct
        self.cfg_slat = cfg_slat

    @torch.inference_mode()
    async def infer_to_ply(
        self,
        image: Image.Image,
        struct_steps: Optional[int] = None,
        slat_steps: Optional[int] = None,
        cfg_struct: Optional[float] = None,
        cfg_slat: Optional[float] = None,
        seed: Optional[int] = None,
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
        buf = io.BytesIO()
        gs.save_ply(buf)
        return buf.getvalue()
