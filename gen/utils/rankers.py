from typing import List, Sequence
from PIL import Image
import numpy as np


def rank_images_quick(
    fgs: Sequence[Image.Image],
    prompt: str | None = None,
    top_k: int = 4,
) -> List[int]:
    """
    Rank RGBA images quickly using only the alpha channel.
    Score = 0.7 * area_frac + 0.3 * (1 - center_dist_norm)
    Returns indices of the top_k images (sorted by descending score).
    """
    scores = []
    for idx, im in enumerate(fgs):
        if im.mode != "RGBA":
            im = im.convert("RGBA")
        a = np.asarray(im.getchannel("A"), dtype=np.float32)  # [H, W] 0..255
        H, W = a.shape

        # Normalize alpha to [0,1]
        m = a / 255.0

        # Area fraction (how much foreground is present)
        area = float(m.mean())  # 0..1

        # Centroid (weighted by alpha)
        total = m.sum() + 1e-6
        ys = np.arange(H, dtype=np.float32)
        xs = np.arange(W, dtype=np.float32)
        cy = float((m * ys[:, None]).sum() / total)  # 0..H-1
        cx = float((m * xs[None, :]).sum() / total)  # 0..W-1

        # Distance from center (normalized 0..1 by half-diagonal)
        dy = cy - (H - 1) / 2.0
        dx = cx - (W - 1) / 2.0
        dist = np.sqrt(dx * dx + dy * dy)
        max_dist = np.sqrt(((H - 1) / 2.0) ** 2 + ((W - 1) / 2.0) ** 2) + 1e-6
        center_score = 1.0 - float(dist / max_dist)  # 1.0 best (centered), 0.0 worst

        # Combine (tweak weights if needed)
        score = 0.7 * area + 0.3 * center_score
        scores.append((score, idx))

    scores.sort(key=lambda t: t[0], reverse=True)
    top = [idx for _, idx in scores[: max(1, top_k)]]
    return top
