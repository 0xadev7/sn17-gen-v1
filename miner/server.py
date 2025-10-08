from __future__ import annotations
import asyncio, base64, io
from time import time
from fastapi import FastAPI, Form
from fastapi.responses import Response, StreamingResponse
from PIL import Image
from loguru import logger

from miner.settings import Config
from miner.state import MinerState

app = FastAPI()
CFG = Config()
STATE: MinerState | None = None


@app.on_event("startup")
def startup():
    global STATE
    STATE = MinerState(CFG)
    logger.info(f"Server up. Port={CFG.port}, GPU={CFG.gpu_id}")


@app.post("/generate")
async def generate(
    prompt: str | None = Form(None),
    image_b64: str | None = Form(None),
):
    """
    Returns binary PLY (Gaussian Splat). MUST return within 30s.
    Returns empty bytes if self-validation fails or timeout occurs.
    """
    assert STATE is not None
    t0 = time()
    try:

        async def _run():
            if image_b64:
                pil = Image.open(io.BytesIO(base64.b64decode(image_b64)))
                return await STATE.image_to_ply(pil)
            else:
                assert prompt is not None and prompt.strip()
                return await STATE.text_to_ply(prompt.strip())

        ply_bytes, score = await asyncio.wait_for(_run(), timeout=CFG.timeout_s)

        elapsed = time() - t0
        mb = len(ply_bytes) / 1e6

        logger.info(f"[generate] score={score:.3f} ply={mb:.1f}MB total={elapsed:.2f}s")
        return Response(ply_bytes, media_type="application/octet-stream")

    except asyncio.TimeoutError:
        logger.warning("[generate] timed out at 30s; returning empty bytes")
        return Response(b"", media_type="application/octet-stream")
