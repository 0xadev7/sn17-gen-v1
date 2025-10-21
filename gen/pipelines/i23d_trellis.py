from __future__ import annotations
import io, torch, gc, threading
from typing import Optional
from PIL import Image
from gen.lib.trellis.pipelines import TrellisImageTo3DPipeline
import contextlib


class TrellisImageTo3D:
    def __init__(
        self,
        device: torch.device,
    ):
        self.device = device
        self.dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

        self.pipe = TrellisImageTo3DPipeline.from_pretrained(
            "microsoft/TRELLIS-image-large",
        )
        self.pipe.to(self.device)

    @torch.inference_mode()
    def infer_to_ply(
        self,
        image: Image.Image,
        struct_steps: Optional[int] = 8,
        slat_steps: Optional[int] = 12,
        cfg_struct: Optional[float] = 7.5,
        cfg_slat: Optional[float] = 3.0,
        seed: Optional[int] = None,
    ) -> bytes:
        """
        Synchronous and device-guarded. We call this inside asyncio.to_thread().
        """

        outputs = self.pipe.run(
            image,
            seed=seed if seed is not None else 1,
            sparse_structure_sampler_params={
                "steps": struct_steps,
                "cfg_strength": cfg_struct,
            },
            slat_sampler_params={
                "steps": slat_steps,
                "cfg_strength": cfg_slat,
            },
        )

        gs = outputs["gaussian"][0]
        buf = io.BytesIO()
        gs.save_ply(buf)
        result = buf.getvalue()

        return result
