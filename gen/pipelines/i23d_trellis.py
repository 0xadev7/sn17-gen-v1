from __future__ import annotations
import io, torch, gc, threading
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
        self.steps_struct = steps_struct
        self.steps_slat = steps_slat
        self.cfg_struct = cfg_struct
        self.cfg_slat = cfg_slat

        self._lock = threading.Lock()
        self._poisoned = False
        self._init_pipe()

    def _init_pipe(self):
        # Dispose previous pipe if present
        if hasattr(self, "pipe"):
            try:
                del self.pipe
                gc.collect()
                if self.device.type == "cuda":
                    with torch.cuda.device(self.device):
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
            except Exception:
                pass

        # (Re)create on the correct GPU and keep it there
        self.pipe = TrellisImageTo3DPipeline.from_pretrained(
            "microsoft/TRELLIS-image-large"
        )
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        self.pipe.to(self.device)
        self._poisoned = False

    @torch.inference_mode()
    def infer_to_ply(
        self,
        image: Image.Image,
        struct_steps: Optional[int] = None,
        slat_steps: Optional[int] = None,
        cfg_struct: Optional[float] = None,
        cfg_slat: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> bytes:
        """
        Synchronous and device-guarded. We call this inside asyncio.to_thread().
        """
        if self._poisoned:
            self._init_pipe()

        with self._lock:
            try:
                if self.device.type == "cuda":
                    torch.cuda.set_device(self.device)
                    torch.cuda.synchronize(self.device)

                outputs = self.pipe.run(
                    image,
                    seed=seed if seed is not None else 1,
                    sparse_structure_sampler_params={
                        "steps": (
                            struct_steps
                            if struct_steps is not None
                            else self.steps_struct
                        ),
                        "cfg_strength": (
                            cfg_struct if cfg_struct is not None else self.cfg_struct
                        ),
                    },
                    slat_sampler_params={
                        "steps": (
                            slat_steps if slat_steps is not None else self.steps_slat
                        ),
                        "cfg_strength": (
                            cfg_slat if cfg_slat is not None else self.cfg_slat
                        ),
                    },
                )

                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)

                gs = outputs["gaussian"][0]
                buf = io.BytesIO()
                gs.save_ply(buf)
                del outputs, gs
                gc.collect()
                if self.device.type == "cuda":
                    with torch.cuda.device(self.device):
                        torch.cuda.empty_cache()
                return buf.getvalue()

            except RuntimeError as e:
                msg = str(e).lower()
                # Treat OOM/illegal access as context poison → reinit on next call
                if (
                    ("illegal memory access" in msg)
                    or ("device-side assert" in msg)
                    or ("out of memory" in msg)
                ):
                    self._poisoned = True
                    if self.device.type == "cuda":
                        with torch.cuda.device(self.device):
                            torch.cuda.empty_cache()
                            torch.cuda.ipc_collect()
                raise
