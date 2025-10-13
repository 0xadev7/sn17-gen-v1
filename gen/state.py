from __future__ import annotations
import base64
import io
from typing import Dict, Tuple, Optional, List
import os
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
from gen.utils.rankers import rank_images_quick


class MinerState:
    def __init__(self, cfg: Config):
        self.cfg = cfg

        # Select devices explicitly (fallback to CPU if needed)
        if torch.cuda.is_available():
            self.t2i_device = torch.device(f"cuda:{cfg.t2i_gpu_id}")
            self.aux_device = torch.device(f"cuda:{cfg.aux_gpu_id}")
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

    # ---------------------------
    # Public APIs with retries
    # ---------------------------
    @torch.inference_mode()
    async def text_to_ply(self, prompt: str) -> Tuple[bytes, float]:
        start = time.time()

        # ---------- Flux: batch N seeds ----------
        N = 6  # try 5–6
        seeds = [random.randint(0, 2**31 - 1) for _ in range(N)]
        steps = self.cfg.t2i_steps
        res = max(768, min(896, self.cfg.t2i_res))
        guidance = self.cfg.t2i_guidance

        imgs = await self.t2i.generate_batch(
            prompt=prompt, seeds=seeds, steps=steps, guidance=guidance, res=res
        )

        # ---------- Batched BG removal ----------
        fgs = self.bg_remover.remove_batch(imgs)

        # ---------- Fast ranker on CPU (pick top-4) ----------
        # Heuristic: prefer larger foreground mask area & centered mass; or use tiny CLIP on CPU.
        ranked = rank_images_quick(fgs, prompt, top_k=4)  # returns indices

        best_score, best_ply = -1.0, b""
        passed_count = 0

        async def validate_async(ply_bytes):
            # non-blocking validation (CPU/network)
            score, passed, _ = await self.validator.validate_text(prompt, ply_bytes)
            return score, (ply_bytes if passed else b""), passed

        # ---------- Trellis sequential (GPU-saturating) ----------
        val_tasks: List[asyncio.Task] = []
        for idx in ranked:
            if (time.time() - start) > (self.time_budget_s or 1e9):
                break
            fg = fgs[idx]

            ply_bytes = await self.trellis_img.infer_to_ply(
                fg,
                struct_steps=self.cfg.trellis_struct_steps,
                slat_steps=self.cfg.trellis_slat_steps,
                cfg_struct=self.cfg.trellis_cfg_struct,
                cfg_slat=self.cfg.trellis_cfg_slat,
                seed=seeds[idx],
            )

            if not ply_bytes:
                continue

            # Launch validation but don't block GPU
            val_tasks.append(asyncio.create_task(validate_async(ply_bytes)))

            # Optional: brief yield to let validation start
            await asyncio.sleep(0)

            # Soft early-stop if we already have 4 validations launched
            if len(val_tasks) >= 4:
                break

        # ---------- Collect validations ----------
        for t in asyncio.as_completed(val_tasks):
            score, maybe_ply, passed = await t
            best_score = max(best_score, score)
            if passed:
                best_ply = maybe_ply
                passed_count += 1
            if best_score >= self.early_stop_score:
                break

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
