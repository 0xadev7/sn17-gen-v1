from __future__ import annotations

import asyncio, os
from time import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import Response, PlainTextResponse
from loguru import logger

from gen.settings import Config
from gen.state import MinerState


CFG = Config()
STATE: Optional[MinerState] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI-recommended startup/shutdown handler.
    Initializes MinerState once; if it fails, we log and keep app up with 503s.
    """
    global STATE
    try:
        STATE = MinerState(CFG)
        logger.info(
            f"Server up. Port={CFG.port}, T2I_GPU={CFG.t2i_gpu_id}, AUX_GPU={CFG.aux_gpu_id}"
        )
    except Exception as e:
        # Do NOT crash app; let /health expose failure
        STATE = None
        logger.exception(f"[lifespan] MinerState init failed: {e}")

    yield

    # If you add pooled resources later, close them here.


app = FastAPI(lifespan=lifespan)


def _require_mode(prompt: Optional[str], image_b64: Optional[str]) -> str:
    if prompt and image_b64:
        raise HTTPException(400, "Provide either 'prompt' or 'image_b64', not both.")
    if not prompt and not image_b64:
        raise HTTPException(400, "Provide one of 'prompt' or 'image_b64'.")
    return "image" if image_b64 else "text"


@app.get("/health", response_class=PlainTextResponse)
async def health():
    # Report whether models loaded
    return "ok" if STATE is not None else "degraded: miner not initialized"


@app.post("/generate/")
async def generate(
    prompt: str | None = Form(None),
    image_b64: str | None = Form(None),
):
    """
    Returns binary PLY (Gaussian Splat). MUST return within CFG.timeout_s.
    Returns empty bytes if self-validation fails or timeout occurs.
    """
    if STATE is None:
        raise HTTPException(503, "Miner not initialized (see logs for details).")

    mode = _require_mode(prompt, image_b64)
    t0 = time()

    async def _run():
        if mode == "image":
            assert image_b64 is not None, "Invalid image input"
            return await STATE.image_to_ply(image_b64)
        else:
            assert prompt is not None and prompt.strip(), "Empty text prompt"
            return await STATE.text_to_ply(prompt.strip())

    timeout_s = getattr(CFG, "timeout_s", 30.0)

    try:
        ply_bytes, score = await asyncio.wait_for(_run(), timeout=timeout_s)
        elapsed = time() - t0
        mb = len(ply_bytes) / 1e6
        logger.info(
            f"[/generate:{mode}] score={score:.3f} ply={mb:.1f}MB total={elapsed:.2f}s"
        )
        return Response(ply_bytes, media_type="application/octet-stream")
    except asyncio.TimeoutError:
        logger.warning(
            f"[/generate:{mode}] timed out at {timeout_s:.0f}s; returning empty bytes"
        )
        return Response(b"", media_type="application/octet-stream")
    except Exception as e:
        # Do not crash; return empty to respect caller contract
        logger.exception(f"[/generate:{mode}] error: {e}")
        return Response(b"", media_type="application/octet-stream")


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    try:
        port = int(os.getenv("PORT", str(CFG.port)))
    except Exception:
        port = CFG.port

    uvicorn.run(app, host=host, port=port, log_level="info")
