from __future__ import annotations
import base64
import io
from typing import Dict, Tuple, Optional, List
from loguru import logger
import asyncio, time, random
import numpy as np
import torch
from PIL import Image
import functools

from gen.settings import Config
from gen.pipelines.t2i_flux import FluxText2Image
from gen.pipelines.bg_birefnet import BiRefNetRemover
from gen.pipelines.i23d_trellis import TrellisImageTo3D
from gen.validators.external_validator import ExternalValidator


class MinerState:
    def __init__(self, cfg: Config):
        self.cfg = cfg

        # Devices (configure via env: T2I_GPU_ID, AUX_GPU_ID)
        if torch.cuda.is_available():
            self.t2i_device = torch.device(f"cuda:{cfg.t2i_gpu_id}")
            self.aux_device = torch.device(f"cuda:{cfg.aux_gpu_id}")
        else:
            self.t2i_device = torch.device("cpu")
            self.aux_device = torch.device("cpu")

        # Pipelines on dedicated GPUs
        self.t2i = FluxText2Image(self.t2i_device)
        self.bg_remover = BiRefNetRemover(self.t2i_device)
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
        self.trellis_max_tries: int = getattr(cfg, "trellis_max_tries", 1)
        self.early_stop_score: float = getattr(
            cfg, "early_stop_score", max(0.0, cfg.vld_threshold)
        )
        self.time_budget_s: Optional[float] = getattr(cfg, "time_budget_s", None)

        # Streaming & concurrency knobs (optional envs; safe defaults)
        self.queue_maxsize: int = int(getattr(cfg, "queue_maxsize", 3))
        self.t2i_concurrency: int = int(
            getattr(cfg, "t2i_concurrency", 1)
        )  # avoid thrash
        self.trellis_concurrency: int = int(
            getattr(cfg, "trellis_concurrency", 1)
        )  # usually 1 per GPU
        self.debug_save: bool = bool(getattr(cfg, "debug_save", False))

        logger.info(
            f"Models loaded. T2I+BG on {self.t2i_device}, Trellis on {self.aux_device}."
        )

    # ---------------------------
    # Internal helpers
    # ---------------------------

    async def _run_blocking(self, fn, *args, **kwargs):
        # Runs sync/blocking work off the event loop
        return await asyncio.to_thread(functools.partial(fn, *args, **kwargs))

    def _t2i_param_sweep(self) -> List[Dict]:
        """
        Sweep T2I parameters across retry attempts.
        Baseline first, then slight jitter on steps and random seeds.
        """
        base_steps = self.cfg.t2i_steps
        base_guidance = self.cfg.t2i_guidance
        base_res = self.cfg.t2i_res

        tries: List[Dict] = []
        for i in range(self.t2i_max_tries):
            # Alternate around baseline minimally to keep SLA
            steps = max(1, base_steps + (i % 2))
            tries.append(
                {
                    "steps": steps,
                    "guidance": base_guidance,
                    "res": base_res,
                    "seed": random.randint(0, 2**31 - 1),
                }
            )
        return tries

    async def _gen_one_image(self, prompt: str, params: dict):
        img = await self._run_blocking(
            self.t2i.generate_sync,
            prompt,
            steps=params["steps"],
            guidance=params["guidance"],
            res=params["res"],
            seed=params["seed"],
        )
        return img, params

    async def _bg_remove_one(self, pil_image: Image.Image) -> Image.Image:
        fg, _ = await self._run_blocking(self.bg_remover.remove, pil_image)
        return fg

    async def _trellis_one(self, pil_image, params: dict):
        seed = params.get("seed")
        try:
            ply_bytes = await self._run_blocking(
                self.trellis_img.infer_to_ply,
                pil_image,
                struct_steps=params["struct_steps"],
                slat_steps=params["slat_steps"],
                cfg_struct=params["cfg_struct"],
                cfg_slat=params["cfg_slat"],
                seed=seed,
            )
        except Exception as e:
            logger.error(f"Trellis error: {e}")
            ply_bytes = b""
        return ply_bytes, params

    def _within_budget(self, start_ts: float) -> bool:
        if self.time_budget_s is None:
            return True
        return (time.time() - start_ts) < self.time_budget_s

    # ---------------------------
    # Producer-Consumer workers
    # ---------------------------
    async def _producer_t2i_bg(self, prompt, q, start_ts, stop_evt):
        sem = asyncio.Semaphore(self.t2i_concurrency)
        tasks = []

        async def _one(params: dict):
            async with sem:
                if stop_evt.is_set() or not self._within_budget(start_ts):
                    return
                t0 = time.time()
                img, iparams = await self._gen_one_image(prompt, params)
                t2i_sec = time.time() - t0
                if stop_evt.is_set() or not self._within_budget(start_ts):
                    return
                t1 = time.time()
                fg = await self._bg_remove_one(img)
                bg_sec = time.time() - t1
                logger.debug(f"[GPU0] T2I {t2i_sec:.2f}s + BG {bg_sec:.2f}s -> queued")
                await q.put((fg, iparams))  # may backpressure if full

        try:
            for p in self._t2i_param_sweep():
                if stop_evt.is_set() or not self._within_budget(start_ts):
                    break
                tasks.append(asyncio.create_task(_one(p)))

            # As soon as the first item is ready, consumer can start pulling it.
            # We still wait here for producer tasks to finish in the background;
            # the consumer is running concurrently on the same loop.
            for t in asyncio.as_completed(tasks):
                await t
        finally:
            await q.put(None)  # sentinel

    async def _consumer_trellis_validate(
        self,
        prompt: str,
        q: asyncio.Queue,
        start_ts: float,
        stop_evt: asyncio.Event,
        best_out: Dict[str, object],
    ):
        """
        GPU1 consumer:
        - pulls foreground images
        - runs trellis (+ optional param sweep)
        - validates
        - tracks best result
        Supports early stop and time budget.
        """
        # Track best locally; then store in best_out
        best_score = -1.0
        best_ply: bytes = b""

        while True:
            if stop_evt.is_set():
                break
            if not self._within_budget(start_ts):
                logger.warning("[GPU1] Budget exhausted; stopping consumer.")
                break

            item = await q.get()
            if item is None:
                # Producer done
                break

            fg, iparams = item

            tparams = {
                "struct_steps": self.cfg.trellis_struct_steps,
                "slat_steps": self.cfg.trellis_slat_steps,
                "cfg_struct": self.cfg.trellis_cfg_struct,
                "cfg_slat": self.cfg.trellis_cfg_slat,
                "seed": random.randint(0, 2**31 - 1),
            }

            if stop_evt.is_set() or not self._within_budget(start_ts):
                break

            t0 = time.time()
            ply_bytes, _ = await self._trellis_one(fg, tparams)
            trellis_sec = time.time() - t0

            if not ply_bytes:
                logger.debug("[GPU1] Empty PLY from Trellis; skipping validation.")
                continue

            v0 = time.time()
            score, passed, _ = await self._run_blocking(
                self.validator.validate_text, prompt, ply_bytes
            )
            vsec = time.time() - v0
            logger.info(
                f"[text] T2I{iparams}|TRELLIS{tparams} -> score={score:.4f}, passed={passed} "
                f"(trellis {trellis_sec:.2f}s, validate {vsec:.2f}s)"
            )

            if score > best_score:
                best_score, best_ply = score, ply_bytes

            if score >= self.early_stop_score:
                logger.info(
                    f"[text] Early-stop at score {score:.4f} >= {self.early_stop_score:.4f}"
                )
                # Save and signal stop
                best_out["ply"] = ply_bytes if passed else b""
                best_out["score"] = score
                stop_evt.set()
                return

            # loop back to the queue for next item

        # Save whatever is best so far (respect threshold)
        final_pass = best_score >= self.cfg.vld_threshold
        best_out["ply"] = best_ply if final_pass else b""
        best_out["score"] = max(0.0, best_score)

    # ---------------------------
    # Public APIs (streaming)
    # ---------------------------

    async def text_to_ply(self, prompt: str) -> Tuple[bytes, float]:
        """
        Streaming two-stage pipeline (GPU0 → GPU1):
          - GPU0: T2I + BG, push to queue as soon as ready
          - GPU1: Trellis + validate, early-stop possible
        """
        start_ts = time.time()
        q: asyncio.Queue = asyncio.Queue(maxsize=self.queue_maxsize)
        stop_evt = asyncio.Event()
        best_out: Dict[str, object] = {}

        # Kick off producer and consumer
        producer_task = asyncio.create_task(
            self._producer_t2i_bg(prompt, q, start_ts, stop_evt)
        )
        consumer_task = asyncio.create_task(
            self._consumer_trellis_validate(prompt, q, start_ts, stop_evt, best_out)
        )

        try:
            await asyncio.gather(producer_task, consumer_task)
        finally:
            # Drain queue if anything left and ensure sentinel consumed
            try:
                while not q.empty():
                    _ = q.get_nowait()
            except Exception:
                pass

        ply_bytes: bytes = best_out.get("ply", b"")  # type: ignore
        score: float = float(best_out.get("score", 0.0))  # type: ignore
        return ply_bytes, score

    async def image_to_ply(self, image_b64) -> Tuple[bytes, float]:
        """
        Image mode stays sequential per image (BG → Trellis → validate),
        since there's a single input. You could stream across multiple inputs at caller level.
        """
        start_ts = time.time()

        try:
            raw = base64.b64decode(image_b64, validate=False)
        except Exception:
            raw = base64.b64decode(image_b64 or "")
        pil_image = Image.open(io.BytesIO(raw)).convert("RGBA")

        # BG removal on GPU0 (as requested)
        fg, _ = self.bg_remover.remove(pil_image)

        best_score = -1.0
        best_ply: bytes = b""

        for tparams in self._trellis_param_sweep():
            if not self._within_budget(start_ts):
                logger.warning("Budget exhausted mid-Trellis(image); stopping.")
                break

            ply_bytes, _ = await self._trellis_one(fg, tparams)
            if not ply_bytes:
                continue

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
