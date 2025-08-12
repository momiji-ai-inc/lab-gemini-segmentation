import io
import base64
import numpy as np
from typing import List, Tuple
from PIL import Image
from models import SegMask

def _as_abs_box(box_2d, W, H) -> Tuple[int, int, int, int]:
    y0, x0, y1, x1 = box_2d
    ay0 = int(np.floor(y0 / 1000.0 * H))
    ax0 = int(np.floor(x0 / 1000.0 * W))
    ay1 = int(np.ceil (y1 / 1000.0 * H))
    ax1 = int(np.ceil (x1 / 1000.0 * W))
    return ay0, ax0, ay1, ax1

def _decode_mask_to_L(mask_b64_png: str):
    if mask_b64_png.startswith("data:image/png;base64,"):
        mask_b64_png = mask_b64_png.split(",", 1)[1]
    buf = io.BytesIO(base64.b64decode(mask_b64_png))

    img = Image.open(buf)
    # パレット画像ならLに変換
    if img.mode == "P":
        img = img.convert("L")
    elif img.mode == "1":
        img = img.convert("L")
    return img.convert("L")


def parse_segmentation_masks(items: List[dict], img_size: Tuple[int,int]) -> List[SegMask]:
    if items is None:
        items = []
    W, H = img_size
    out: List[SegMask] = []
    for idx, it in enumerate(items):
        if not isinstance(it, dict) or "box_2d" not in it or "mask" not in it:
            continue
        ay0, ax0, ay1, ax1 = _as_abs_box(it["box_2d"], W, H)
        if not (0 <= ax0 < ax1 <= W and 0 <= ay0 < ay1 <= H):
            continue
        m_L = _decode_mask_to_L(it["mask"])
        if m_L is None:
            continue
        m = m_L.resize((ax1-ax0, ay1-ay0), Image.Resampling.BICUBIC)
        full = Image.new("L", (W, H), color=0)
        full.paste(m, (ax0, ay0))
        label = (it.get("label") or f"item_{idx}").strip()
        out.append(SegMask(y0=ay0, x0=ax0, y1=ay1, x1=ax1, label=label, full_mask_L=full))
    return out
