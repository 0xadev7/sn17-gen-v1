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
        steps_struct: int,
        steps_slat: int,
        cfg_struct: float,
        cfg_slat: float,
        dtype: torch.dtype = torch.bfloat16,  # or torch.float16
        warmup: bool = True,
    ):
        self.device = device
        self.steps_struct = steps_struct
        self.steps_slat = steps_slat
        self.cfg_struct = cfg_struct
        self.cfg_slat = cfg_slat
        self.dtype = dtype
        self.warmup = warmup

        self._lock = threading.Lock()
        self._poisoned = False
        self._init_pipe()

    def _empty(self):
        gc.collect()
        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                # Avoid ipc_collect() unless you actually share tensors via CUDA IPC.
                # torch.cuda.ipc_collect()

    def _init_pipe(self):
        # Dispose previous pipe if present
        if hasattr(self, "pipe"):
            try:
                del self.pipe
            except Exception:
                pass
            self._empty()

        if self.device.type == "cuda":
            # pin this process to the intended device
            torch.cuda.set_device(self.device)

        # (Re)create on the correct GPU and keep it there
        self.pipe = TrellisImageTo3DPipeline.from_pretrained(
            "microsoft/TRELLIS-image-large"
        )
        # Move + set dtype consistently to reduce mixed-precision fragmentation
        self.pipe.to(self.device)
        try:
            # Not all modules respect a global dtype; guard it.
            for p in self.pipe.parameters():
                p.data = p.data.to(self.dtype, non_blocking=True)
        except Exception:
            pass

        # Optional: a tiny warmup run to “shape” the allocator once, not mid-request.
        if self.warmup:
            try:
                img = Image.new("RGB", (256, 256), (0, 0, 0))
                _ = self.pipe.run(
                    img,
                    seed=1,
                    sparse_structure_sampler_params={"steps": 1, "cfg_strength": 1.0},
                    slat_sampler_params={"steps": 1, "cfg_strength": 1.0},
                )
                self._empty()
            except Exception:
                # Warmup failure shouldn't prevent service
                pass

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
            # Use a dedicated stream to avoid cross-op interleaving on default stream
            stream = None
            try:
                if self.device.type == "cuda":
                    torch.cuda.set_device(self.device)
                    stream = torch.cuda.Stream(device=self.device)
                    torch.cuda.synchronize(self.device)
                    stream.wait_stream(torch.cuda.current_stream(self.device))

                ctx = (
                    torch.cuda.stream(stream)
                    if stream is not None
                    else contextlib.nullcontext()
                )
                with ctx:
                    outputs = self.pipe.run(
                        image.convert("RGB"),  # enforce RGB
                        seed=seed if seed is not None else 1,
                        sparse_structure_sampler_params={
                            "steps": (
                                struct_steps
                                if struct_steps is not None
                                else self.steps_struct
                            ),
                            "cfg_strength": (
                                cfg_struct
                                if cfg_struct is not None
                                else self.cfg_struct
                            ),
                        },
                        slat_sampler_params={
                            "steps": (
                                slat_steps
                                if slat_steps is not None
                                else self.steps_slat
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
                    result = buf.getvalue()

                # Clean-up
                del outputs, gs
                self._empty()
                return result

            except RuntimeError as e:
                msg = str(e).lower()
                # Treat allocator assert / OOM / device faults as poison → reinit next call
                if (
                    "c10/cuda/cudacachingallocator.cpp" in msg
                    or "!block->expandable_segment_" in msg
                    or "illegal memory access" in msg
                    or "device-side assert" in msg
                    or "out of memory" in msg
                ):
                    self._poisoned = True
                    self._empty()
                raise
