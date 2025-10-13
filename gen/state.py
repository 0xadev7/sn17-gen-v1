from __future__ import annotations
import base64
import io
import contextlib
from typing import Dict, Tuple, Optional, List
from loguru import logger
import asyncio, time, random
import numpy as np
import torch
from PIL import Image

from gen.settings import Config
from gen.pipelines.t2i_flux import FluxText2Image
from gen.pipelines.bg_birefnet import BiRefNetRemover
from gen.pipelines.i23d_trellis import TrellisImageTo3D
from gen.validators.external_validator import ExternalValidator


class MinerState:
    def __init__(self, cfg: Config):
        self.cfg = cfg

        # Select devices explicitly (fallback to CPU if needed)
        if torch.cuda.is_available():
            self.t2i_device = torch.device(f"cuda:{cfg.t2i_gpu_id}")
            self.aux_device = torch.device(f"cuda:{cfg.aux_gpu_id}")
            torch.backends.cudnn.benchmark = True
        else:
            self.t2i_device = torch.device("cpu")
            self.aux_device = torch.device("cpu")

        # Pipelines on dedicated GPUs
        self.t2i = FluxText2Image(self.t2i_device)
        self.bg_remover = BiRefNetRemover(self.aux_device)
        self.trellis_img = TrellisImageTo3D(
            self.aux_device,
            cfg.trellis_struct_steps,
            cfg.trellis_slat_steps,
            cfg.trellis_cfg_struct,
            cfg.trellis_cfg_slat,
        )

        # Validators (HTTP)
        self.validator = ExternalValidator(
            cfg.validator_url_txt, cfg.validator_url_img, cfg.vld_threshold
        )

        # Retry/budget knobs...
        self.t2i_max_tries: int = getattr(cfg, "t2i_max_tries", 3)
        self.trellis_max_tries: int = getattr(cfg, "trellis_max_tries", 2)
        self.early_stop_score: float = getattr(
            cfg, "early_stop_score", max(0.0, cfg.vld_threshold)
        )
        self.time_budget_s: Optional[float] = getattr(cfg, "time_budget_s", None)

        logger.info(
            f"Models loaded. T2I on {self.t2i_device}, BiRefNet+Trellis on {self.aux_device}."
        )

    # ---------------------------
    # Internal helpers
    # ---------------------------

    def _t2i_param_sweep(self) -> List[Dict]:
        """
        Sweep T2I parameters across retry attempts.
        Ensure baseline is tried first, then jittered variants.
        """
        base_steps = self.cfg.t2i_steps
        base_guidance = self.cfg.t2i_guidance
        base_res = self.cfg.t2i_res

        tries: List[Dict] = []
        for i in range(self.t2i_max_tries):
            delta_steps = i % 2

            if i % 2 == 1:
                steps = base_steps + delta_steps
            else:
                steps = base_steps - delta_steps

            tries.append(
                {
                    "steps": max(1, steps),
                    "guidance": base_guidance,
                    "res": base_res,
                    "seed": random.randint(0, 2**31 - 1),
                }
            )
        return tries

    def _trellis_param_sweep(self) -> List[Dict]:
        """
        Sweep Trellis parameters for retries.
        Baseline first, then jitter structural + cfg values.
        """
        bs = self.cfg.trellis_struct_steps
        bl = self.cfg.trellis_slat_steps
        bc = self.cfg.trellis_cfg_struct
        bd = self.cfg.trellis_cfg_slat

        tries: List[Dict] = []
        for i in range(self.trellis_max_tries):
            if i == 0:
                tries.append(
                    {
                        "struct_steps": bs,
                        "slat_steps": bl,
                        "cfg_struct": bc,
                        "cfg_slat": bd,
                        "seed": random.randint(0, 2**31 - 1),
                    }
                )
            else:
                # jitter scale: ±15% steps, ±0.4 cfg
                frac_steps = 0.15
                frac_cfg = 0.4

                delta_ss = int(
                    round(bs * frac_steps * ((i) / (self.trellis_max_tries - 1)))
                )
                delta_sl = int(
                    round(bl * frac_steps * ((i) / (self.trellis_max_tries - 1)))
                )

                if i % 2 == 1:
                    struct_steps = bs + delta_ss
                    slat_steps = bl + delta_sl
                else:
                    struct_steps = bs - delta_ss
                    slat_steps = bl - delta_sl

                delta_c_str = frac_cfg * ((i) / (self.trellis_max_tries - 1))
                delta_c_slat = frac_cfg * ((i) / (self.trellis_max_tries - 1))

                if i % 2 == 1:
                    cfg_struct = bc + delta_c_str
                    cfg_slat = bd + delta_c_slat
                else:
                    cfg_struct = bc - delta_c_str
                    cfg_slat = bd - delta_c_slat

                tries.append(
                    {
                        "struct_steps": max(1, struct_steps),
                        "slat_steps": max(1, slat_steps),
                        "cfg_struct": cfg_struct,
                        "cfg_slat": cfg_slat,
                        "seed": random.randint(0, 2**31 - 1),
                    }
                )
        return tries

    async def _gen_one_image(self, prompt: str, params: dict):
        img = await self.t2i.generate(
            prompt,
            steps=params["steps"],
            guidance=params["guidance"],
            res=params["res"],
            seed=params["seed"],
        )

        return img, params

    async def _trellis_one(self, pil_image, params: dict):
        seed = params.get("seed")
        try:
            ply_bytes = await self.trellis_img.infer_to_ply(
                pil_image,
                struct_steps=params["struct_steps"],
                slat_steps=params["slat_steps"],
                cfg_struct=params["cfg_struct"],
                cfg_slat=params["cfg_slat"],
                seed=seed,
            )
        except Exception as e:
            logger.error("Trellis error: ", e)
            ply_bytes = b""
        return ply_bytes, params

    def _within_budget(self, start_ts: float) -> bool:
        if self.time_budget_s is None:
            return True
        return (time.time() - start_ts) < self.time_budget_s

    def _now(self) -> float:
        return time.time()

    def _deadline(self, start_ts: float) -> float:
        if self.time_budget_s is None:
            # default to very large window
            return start_ts + 10_000
        return start_ts + self.time_budget_s

    def _time_left(self, deadline: float) -> float:
        return max(0.0, deadline - self._now())

    # Optional: mini-batcher for T2I if you later want to chunk seeds; here we just
    # limit concurrency via semaphore.
    async def _generate_batch(
        self, prompt: str, param_list: List[Dict], sem: asyncio.Semaphore
    ):
        async def one(p):
            async with sem:
                return await self._gen_one_image(prompt, p)

        return await asyncio.gather(*[one(p) for p in param_list])

    # ---------------------------
    # Public APIs with retries
    # ---------------------------

    async def text_to_ply(self, prompt: str) -> Tuple[bytes, float]:
        start_ts = time.time()

        # 1) Generate multiple images (fan-out)
        t2i_params = self._t2i_param_sweep()
        image_tasks = [self._gen_one_image(prompt, p) for p in t2i_params]

        # If you need stricter SLA, you can limit concurrency here.
        t2i_results: List[Tuple] = await asyncio.gather(*image_tasks)

        best_score = -1.0
        best_ply: bytes = b""

        for img, iparams in t2i_results:
            if not self._within_budget(start_ts):
                logger.warning("Budget exhausted after T2I; stopping.")
                break

            # 2) Background removal
            fg, _ = self.bg_remover.remove(img)

            # 3) Multiple Trellis tries (usually sequential to avoid GPU thrash)
            for tparams in self._trellis_param_sweep():
                if not self._within_budget(start_ts):
                    logger.warning("Budget exhausted mid-Trellis; stopping.")
                    break

                ply_bytes, _ = await self._trellis_one(fg, tparams)

                if len(ply_bytes) == 0:
                    continue

                # 4) Validate
                score, passed, _ = await self.validator.validate_text(prompt, ply_bytes)
                logger.info(
                    f"[text] T2I{iparams}|TRELLIS{tparams} -> score={score:.4f}, passed={passed}"
                )

                if score > best_score:
                    best_score, best_ply = score, ply_bytes

                # Early stop if we’ve cleared a high bar
                if score >= self.early_stop_score:
                    logger.info(
                        f"[text] Early-stop: score {score:.4f} >= {self.early_stop_score:.4f}"
                    )
                    return (ply_bytes if passed else b""), score

        # Return best seen (respect validator threshold)
        final_pass = best_score >= self.cfg.vld_threshold
        return (best_ply if final_pass else b""), max(0.0, best_score)

    async def text_to_ply_parallel(
        self,
        prompt: str,
    ) -> Tuple[bytes, float]:
        """
        Overlapped pipeline:
          T2I (producer on t2i_device) -> queue -> workers run (BG -> Trellis -> validate)
        """
        start_ts = self._now()
        deadline = self._deadline(start_ts)

        # Queues & state
        q: asyncio.Queue = asyncio.Queue(maxsize=self.cfg.trellis_workers * 2)
        stop_event = asyncio.Event()

        best_score = -1.0
        best_ply: bytes = b""

        # --- Producer: generate images with varied steps/seed and push to queue ----
        t2i_params = self._t2i_param_sweep()

        async def producer():
            sem = asyncio.Semaphore(self.cfg.t2i_concurrency)
            # Launch T2I tasks but push to queue as each finish (streaming)
            produce_tasks = [
                asyncio.create_task(self._gen_one_image(prompt, p)) for p in t2i_params
            ]

            try:
                for coro in asyncio.as_completed(produce_tasks):
                    if stop_event.is_set() or self._time_left(deadline) <= 0:
                        break
                    try:
                        img, iparams = await coro
                    except Exception as e:
                        logger.error(f"T2I failed: {e}")
                        continue
                    await q.put((img, iparams))
            finally:
                # Signal no more items
                for _ in range(self.cfg.trellis_workers):
                    await q.put(None)

        # --- Worker: pull images and try a few Trellis variants each ---------------
        async def worker(worker_id: int):
            nonlocal best_score, best_ply
            while not stop_event.is_set():
                # Budget check
                if self._time_left(deadline) <= 0:
                    break

                item = await q.get()
                if item is None:
                    q.task_done()
                    break

                img, iparams = item
                try:
                    # 1) BG removal (fast on aux_device)
                    fg, _ = self.bg_remover.remove(img)

                    # 2) Try a *few* Trellis variants for this image
                    #    We cap per-image tries to keep throughput high.
                    tparams_all = self._trellis_param_sweep()
                    tparams_subset = tparams_all[
                        : max(1, self.cfg.trellis_tries_per_image)
                    ]

                    for tparams in tparams_subset:
                        if stop_event.is_set() or self._time_left(deadline) <= 0:
                            break

                        ply_bytes, _ = await self._trellis_one(fg, tparams)
                        if not ply_bytes:
                            continue

                        # 3) Validate (HTTP: overlap friendly)
                        score, passed, _ = await self.validator.validate_text(
                            prompt, ply_bytes
                        )
                        logger.info(
                            f"[text|W{worker_id}] T2I{iparams}|TRELLIS{tparams} -> {score:.4f} (passed={passed})"
                        )

                        if score > best_score:
                            best_score, best_ply = score, (
                                ply_bytes if passed else best_ply
                            )

                        # Early stop
                        if score >= self.early_stop_score:
                            logger.info(
                                f"[text] Early-stop: {score:.4f} >= {self.early_stop_score:.4f}"
                            )
                            stop_event.set()
                            break
                except Exception as e:
                    logger.error(f"Worker {worker_id} error: {e}")
                finally:
                    q.task_done()

        # Run producer+workers
        workers = [
            asyncio.create_task(worker(i)) for i in range(self.cfg.trellis_workers)
        ]
        prod = asyncio.create_task(producer())

        try:
            await asyncio.wait([prod, *workers], return_when=asyncio.ALL_COMPLETED)
        finally:
            stop_event.set()
            # Drain queue if anything left
            with contextlib.suppress(Exception):
                while not q.empty():
                    q.get_nowait()
                    q.task_done()

        final_pass = best_score >= self.cfg.vld_threshold
        return (best_ply if final_pass else b""), max(0.0, best_score)

    async def image_to_ply(self, image_b64) -> Tuple[bytes, float]:
        start_ts = time.time()

        try:
            raw = base64.b64decode(
                image_b64, validate=False
            )  # tolerate URL-safe/non-padded
        except Exception:
            # fallback without strict validation
            raw = base64.b64decode(image_b64 or "")
        pil_image = Image.open(io.BytesIO(raw)).convert("RGBA")

        # 1) BG removal (once)
        fg, _ = self.bg_remover.remove(pil_image)

        best_score = -1.0
        best_ply: bytes = b""

        # 2) Multiple Trellis tries
        for tparams in self._trellis_param_sweep():
            if not self._within_budget(start_ts):
                logger.warning("Budget exhausted mid-Trellis(image); stopping.")
                break

            ply_bytes, _ = await self._trellis_one(fg, tparams)

            if len(ply_bytes) == 0:
                continue

            # 3) Validate
            score, passed, _ = await self.validator.validate_image(image_b64, ply_bytes)
            logger.info(
                f"[image] TRELLIS{tparams} -> score={score:.4f}, passed={passed}"
            )

            if score > best_score:
                best_score, best_ply = score, ply_bytes

            if score >= self.early_stop_score:
                logger.info(
                    f"[image] Early-stop: score {score:.4f} >= {self.early_stop_score:.4f}"
                )
                return (ply_bytes if passed else b""), score

        final_pass = best_score >= self.cfg.vld_threshold
        return (best_ply if final_pass else b""), max(0.0, best_score)
