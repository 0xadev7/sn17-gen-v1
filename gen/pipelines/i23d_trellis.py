from __future__ import annotations
import io, torch
from typing import Optional
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
        # If Trellis supports dtype/device kwargs, prefer those; otherwise keep .to()
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
            # if self.device.type == "cuda":
            #     autocast_ctx = torch.autocast(device_type="cuda", dtype=self.dtype)
            # else:
            #     # no-op context manager
            #     class _Noop:
            #         def __enter__(self):
            #             pass

            #         def __exit__(self, *a):
            #             pass

            #     autocast_ctx = _Noop()

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

            try:
                # Force materialization on CPU, so GPU refs don’t linger.
                gs = outputs["gaussian"][0]  # Trellis object
                buf = io.BytesIO()
                gs.save_ply(buf)  # should not require GPU after this line
                result = buf.getvalue()
            finally:
                # Defensive cleanup of large refs
                try:
                    del gs
                except Exception:
                    pass
                try:
                    del outputs
                except Exception:
                    pass
                try:
                    buf.close()
                except Exception:
                    pass

            return result
