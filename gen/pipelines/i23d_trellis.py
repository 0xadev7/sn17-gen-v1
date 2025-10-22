from __future__ import annotations
import io, torch
from typing import Optional
from contextlib import nullcontext
from PIL import Image

from gen.lib.trellis.pipelines import TrellisImageTo3DPipeline
from gen.utils.vram import vram_guard


class TrellisImageTo3D:
    def __init__(self, device: torch.device):
        self.device = device
        self.dtype = torch.float16 if device.type == "cuda" else torch.float32

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

        with vram_guard():
            # Optional: autocast on CUDA for memory/perf
            # autocast_ctx = (
            #     torch.cuda.amp.autocast(dtype=self.dtype)
            #     if self.device.type == "cuda"
            #     else nullcontext()
            # )

            outputs = None
            gs = None
            buf = None
            result = None

            try:
                # with autocast_ctx:
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

                # Extract, serialize to PLY (in-memory)
                gs = outputs["gaussian"][0]  # Trellis object
                buf = io.BytesIO()
                gs.save_ply(buf)  # should no longer require GPU after this
                result = buf.getvalue()

                return result

            finally:
                # Defensive cleanup: only touch objects that exist
                try:
                    if buf is not None:
                        buf.close()
                except Exception:
                    pass
                try:
                    del gs
                except Exception:
                    pass
                try:
                    del outputs
                except Exception:
                    pass

                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
