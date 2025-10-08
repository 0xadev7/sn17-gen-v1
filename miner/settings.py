from __future__ import annotations
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    port: int = int(os.getenv("PORT", 7000))
    gpu_id: int = int(os.getenv("GPU_ID", 0))
    timeout_s: float = float(os.getenv("TIMEOUT_S", 30.0))

    # Validation
    validator_url_txt: str = os.getenv(
        "VALIDATOR_TXT_URL", "http://localhost:8094/validate_txt_to_3d_ply/"
    )
    validator_url_img: str = os.getenv(
        "VALIDATOR_IMG_URL", "http://localhost:8094/validate_img_to_3d_ply/"
    )
    vld_threshold: float = float(os.getenv("VALIDATION_THRESHOLD", 0.7))

    # Text-to-2D parameters
    t2i_steps: int = int(os.getenv("T2I_STEPS", 2))
    t2i_guidance: float = float(os.getenv("T2I_GUIDANCE", 3.5))
    t2i_res: int = int(os.getenv("T2I_RES", 704))

    # Trellis parameters
    trellis_struct_steps: int = int(os.getenv("TRELLIS_STRUCT_STEPS", 6))
    trellis_slat_steps: int = int(os.getenv("TRELLIS_SLAT_STEPS", 6))
    trellis_cfg_struct: float = float(os.getenv("TRELLIS_CFG_STRUCT", 7.5))
    trellis_cfg_slat: float = float(os.getenv("TRELLIS_CFG_SLAT", 3.0))
    trellis_max_gaussians: int = int(os.getenv("TRELLIS_MAX_GS", 600_000))
    trellis_target_mb: int = int(os.getenv("TRELLIS_TARGET_MB", 40))
