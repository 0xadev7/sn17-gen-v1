from __future__ import annotations
import base64
import io
from typing import Dict, Tuple, Optional, List
from loguru import logger
import time, random
import torch
from PIL import Image

from gen.settings import Config
from gen.pipelines.i2i_sd35 import SD35Image2Image
from gen.pipelines.t2i_sd35 import SD35Text2Image
from gen.pipelines.bg_birefnet import BiRefNetRemover
from gen.pipelines.i23d_trellis import TrellisImageTo3D
from gen.validators.external_validator import ExternalValidator
from gen.utils.vram import vram_guard


class MinerState:
    def __init__(self, cfg: Config):
        self.cfg = cfg

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Pipelines pinned to devices
        self.t2i = SD35Text2Image(self.device)
        # self.i2i = SD35Image2Image(self.device)
        self.bg_remover = BiRefNetRemover(self.device)
        self.trellis_img = TrellisImageTo3D(self.device)

        # External validator
        self.validator = ExternalValidator(
            cfg.validator_url_txt, cfg.validator_url_img, cfg.vld_threshold
        )

        # Knobs
        self.t2i_max_tries: int = getattr(cfg, "t2i_max_tries", 3)
        self.trellis_max_tries: int = getattr(cfg, "trellis_max_tries", 1)
        self.early_stop_score: float = getattr(
            cfg, "early_stop_score", max(0.0, cfg.vld_threshold)
        )
        self.time_budget_s: Optional[float] = getattr(cfg, "time_budget_s", None)

        self.debug_save: bool = bool(getattr(cfg, "debug_save", False))

        logger.info("Models loaded.")

    # ---------------------------
    # Helpers
    # ---------------------------

    def _t2i_param_sweep(self) -> List[Dict]:
        base_steps = self.cfg.t2i_steps
        base_guidance = self.cfg.t2i_guidance
        base_res = self.cfg.t2i_res

        tries: List[Dict] = []
        for i in range(self.t2i_max_tries):
            steps = max(1, base_steps + (i % 2))  # 2/3 steps alternation
            tries.append(
                {
                    "steps": steps,
                    "guidance": base_guidance,
                    "res": base_res,
                    "seed": random.randint(0, 2**31 - 1),
                }
            )
        return tries

    def _trellis_param_sweep(self) -> List[Dict]:
        params = []
        for _ in range(max(1, self.trellis_max_tries)):
            params.append(
                {
                    "struct_steps": self.cfg.trellis_struct_steps,
                    "slat_steps": self.cfg.trellis_slat_steps,
                    "cfg_struct": self.cfg.trellis_cfg_struct,
                    "cfg_slat": self.cfg.trellis_cfg_slat,
                    "seed": random.randint(0, 2**31 - 1),
                }
            )
        return params

    async def _gen_one_image(self, prompt: str, params: dict):
        return self.t2i.generate(
            prompt,
            steps=params["steps"],
            guidance=params["guidance"],
            res=params["res"],
            seed=params["seed"],
        )

    async def _bg_remove_one(self, pil_image: Image.Image):
        fg, _ = self.bg_remover.remove(pil_image)
        return fg

    async def _trellis_one(self, pil_image, params: dict):
        return self.trellis_img.infer_to_ply(
            pil_image,
            struct_steps=params["struct_steps"],
            slat_steps=params["slat_steps"],
            cfg_struct=params["cfg_struct"],
            cfg_slat=params["cfg_slat"],
            seed=params.get("seed"),
        )

    def _within_budget(self, start_ts: float) -> bool:
        if self.time_budget_s is None:
            return True
        return (time.time() - start_ts) < self.time_budget_s

    # ---------------------------
    # Public APIs
    # ---------------------------

    async def text_to_ply(self, prompt: str) -> Tuple[bytes, float]:
        start_ts = time.time()
        best_score = -1.0
        best_ply: bytes = b""

        for iparams in self._t2i_param_sweep():
            if not self._within_budget(start_ts):
                logger.warning("Budget exhausted mid-T2I; stopping.")
                break

            with vram_guard():
                t0 = time.time()
                img = await self._gen_one_image(prompt, iparams)
                t2i_sec = time.time() - t0
                logger.debug(f"T2I: {t2i_sec:.2f}s")

            if not self._within_budget(start_ts):
                logger.warning("Budget exhausted mid-BG Removal; stopping.")
                # clean up
                try:
                    img.close()
                except Exception:
                    pass
                del img
                break

            with vram_guard():
                t0 = time.time()
                fg = await self._bg_remove_one(img)
                # PIL image no longer needed
                try:
                    img.close()
                except Exception:
                    pass
                del img
                bg_sec = time.time() - t0
                logger.debug(f"BG: {bg_sec:.2f}s")

            for tparams in self._trellis_param_sweep():
                if not self._within_budget(start_ts):
                    logger.warning("Budget exhausted mid-Trellis; stopping.")
                    break

                with vram_guard(ipc_collect=True):
                    t0 = time.time()
                    ply_bytes = await self._trellis_one(fg, tparams)
                    trellis_sec = time.time() - t0
                    logger.debug(f"Trellis: {trellis_sec:.2f}s")

                if not ply_bytes:
                    fg.save(
                        f"out/error_{prompt.replace(' ', '_')}_{iparams['seed']}.png"
                    )
                    continue

                if not self._within_budget(start_ts):
                    logger.warning("Budget exhausted mid-Validation; stopping.")
                    break

                t0 = time.time()
                score, passed, _ = await self.validator.validate_text(prompt, ply_bytes)
                validate_sec = time.time() - t0
                logger.info(
                    f"Validate: score={score:.4f}, passed={passed}, {validate_sec:.2f}s"
                )

                if not passed:
                    fg.save(
                        f"out/error_{prompt.replace(' ', '_')}_{iparams['seed']}.png"
                    )

                if score > best_score:
                    best_score, best_ply = score, ply_bytes

                if score >= self.early_stop_score:
                    logger.info(
                        f"[text] Early-stop: score {score:.4f} >= {self.early_stop_score:.4f}"
                    )
                    # free fg now that we’re done
                    try:
                        fg.close()
                    except Exception:
                        pass
                    del fg
                    return (ply_bytes if passed else b""), score

            # free between outer t2i sweeps
            try:
                fg.close()
            except Exception:
                pass
            del fg

        final_pass = best_score >= self.cfg.vld_threshold
        return (best_ply if final_pass else b""), max(0.0, best_score)

    async def image_to_ply(self, image_b64) -> Tuple[bytes, float]:
        start_ts = time.time()
        best_score = -1.0
        best_ply: bytes = b""

        # decode → PIL
        try:
            raw = base64.b64decode(image_b64, validate=False)
        except Exception:
            raw = base64.b64decode(image_b64 or "")
        pil_image = Image.open(io.BytesIO(raw)).convert("RGBA")

        for iparams in self._t2i_param_sweep():
            if not self._within_budget(start_ts):
                logger.warning("Budget exhausted mid-I2I; stopping.")
                break

            with vram_guard():
                t0 = time.time()
                # img = await self.i2i.generate(
                #     pil_image,
                #     steps=iparams["steps"],
                #     guidance=iparams["guidance"],
                #     res=iparams["res"],
                #     seed=iparams["seed"],
                # )
                img = pil_image
                i2i_sec = time.time() - t0
                logger.debug(f"I2I: {i2i_sec:.2f}s")

            if not self._within_budget(start_ts):
                logger.warning("Budget exhausted mid-BG Removal; stopping.")
                # clean up
                try:
                    img.close()
                except Exception:
                    pass
                del img
                break

            with vram_guard():
                t0 = time.time()
                fg = await self._bg_remove_one(img)
                # PIL image no longer needed
                try:
                    img.close()
                except Exception:
                    pass
                del img
                bg_sec = time.time() - t0
                logger.debug(f"BG: {bg_sec:.2f}s")

            for tparams in self._trellis_param_sweep():
                if not self._within_budget(start_ts):
                    logger.warning("Budget exhausted mid-Trellis; stopping.")
                    break

                with vram_guard(ipc_collect=True):
                    t0 = time.time()
                    ply_bytes = await self._trellis_one(fg, tparams)
                    trellis_sec = time.time() - t0
                    logger.debug(f"Trellis: {trellis_sec:.2f}s")

                if not ply_bytes:
                    fg.save(f"out/error_{random.randint(0, 2**31 - 1)}.png")
                    continue

                if not self._within_budget(start_ts):
                    logger.warning("Budget exhausted mid-Validation; stopping.")
                    break

                t0 = time.time()
                score, passed, _ = await self.validator.validate_image(
                    image_b64, ply_bytes
                )
                validate_sec = time.time() - t0
                logger.info(
                    f"Validate: score={score:.4f}, passed={passed}, {validate_sec:.2f}s"
                )

                if not passed:
                    fg.save(f"out/error_{random.randint(0, 2**31 - 1)}.png")

                if score > best_score:
                    best_score, best_ply = score, ply_bytes

                if score >= self.early_stop_score:
                    logger.info(
                        f"[text] Early-stop: score {score:.4f} >= {self.early_stop_score:.4f}"
                    )
                    # free fg now that we’re done
                    try:
                        fg.close()
                    except Exception:
                        pass
                    del fg
                    return (ply_bytes if passed else b""), score

            # free between outer t2i sweeps
            try:
                fg.close()
            except Exception:
                pass
            del fg

        final_pass = best_score >= self.cfg.vld_threshold
        return (best_ply if final_pass else b""), max(0.0, best_score)

    def close(self):
        # drop big refs
        for obj in [self.t2i, self.bg_remover, self.trellis_img]:
            try:
                del obj
            except Exception:
                pass
        import gc, torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
