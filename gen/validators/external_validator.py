from __future__ import annotations
import base64
from typing import Optional, Tuple
import httpx
from loguru import logger


class ExternalValidator:
    def __init__(self, url_txt: str, url_img: str, threshold: float = 0.7):
        self.url_txt = url_txt
        self.url_img = url_img
        self.threshold = threshold

    async def _call_external_validator(
        self,
        *,
        mode: str,
        prompt: Optional[str],
        prompt_image_b64: Optional[str],
        ply_bytes: bytes,
    ) -> Tuple[float, bool, dict]:
        """
        Sends base64 PLY to the external validator and returns (score, passed, raw_json).
        Uses the `score` field of ValidationResponse directly.
        """
        url = self.url_txt if mode == "text" else self.url_img
        payload = {
            "prompt": prompt,
            "prompt_image": prompt_image_b64,
            "data": base64.b64encode(ply_bytes).decode("utf-8"),
            "compression": 0,
            "generate_single_preview": False,
            "generate_grid_preview": False,
            "preview_score_threshold": float(self.threshold),
        }

        try:
            timeout = httpx.Timeout(connect=3.0, read=6.0, write=3.0, pool=3.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                js = resp.json()
        except Exception as e:
            logger.warning(f"[validator:{mode}] error: {e}")
            return 0.0, False, {"error": str(e)}
        try:
            score = float(js.get("score", 0.0))
        except Exception:
            score = 0.0
        passed = score >= float(self.threshold)
        return score, passed, js

    async def validate_text(
        self, prompt: str, ply_bytes: bytes
    ) -> Tuple[float, bool, dict]:
        return await self._call_external_validator(
            mode="text", prompt=prompt, prompt_image_b64=None, ply_bytes=ply_bytes
        )

    async def validate_image(
        self, img_b64: str, ply_bytes: bytes
    ) -> Tuple[float, bool, dict]:
        return await self._call_external_validator(
            mode="image", prompt=None, prompt_image_b64=img_b64, ply_bytes=ply_bytes
        )
