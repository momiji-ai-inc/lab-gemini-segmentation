import io
import base64
import numpy as np
from typing import List, Tuple
from PIL import Image
from models import SegMask

def _as_abs_box(box_2d, img_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """正規化座標(0-1000)を絶対座標(ピクセル)に変換"""
    width, height = img_size
    y0, x0, y1, x1 = box_2d
    abs_y0 = int(np.floor(y0 / 1000.0 * height))
    abs_x0 = int(np.floor(x0 / 1000.0 * width))
    abs_y1 = int(np.ceil(y1 / 1000.0 * height))
    abs_x1 = int(np.ceil(x1 / 1000.0 * width))
    return abs_y0, abs_x0, abs_y1, abs_x1

def _decode_mask_to_L(mask_b64_png: str):
    """base64エンコードされたPNGマスクをグレースケール画像に変換"""
    if mask_b64_png.startswith("data:image/png;base64,"):
        mask_b64_png = mask_b64_png.split(",", 1)[1]
    # base64のパディングを補う
    missing_padding = len(mask_b64_png) % 4
    if missing_padding:
        mask_b64_png += '=' * (4 - missing_padding)
    buffer = io.BytesIO(base64.b64decode(mask_b64_png))

    mask_image = Image.open(buffer)
    # パレット画像ならLに変換
    if mask_image.mode == "P":
        mask_image = mask_image.convert("L")
    elif mask_image.mode == "1":
        mask_image = mask_image.convert("L")
    return mask_image.convert("L")


def parse_segmentation_masks(items: List[dict], img_size: Tuple[int,int]) -> List[SegMask]:
    """APIレスポンスのJSONからSegMaskオブジェクトのリストを生成"""
    if items is None:
        items = []
    width, height = img_size
    result: List[SegMask] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict) or "box_2d" not in item or "mask" not in item:
            continue
        abs_y0, abs_x0, abs_y1, abs_x1 = _as_abs_box(item["box_2d"], img_size)
        if not (0 <= abs_x0 < abs_x1 <= width and 0 <= abs_y0 < abs_y1 <= height):
            continue
        mask_grayscale = _decode_mask_to_L(item["mask"])
        if mask_grayscale is None:
            continue
        resized_mask = mask_grayscale.resize((abs_x1-abs_x0, abs_y1-abs_y0), Image.Resampling.BICUBIC)
        full_mask = Image.new("L", (width, height), color=0)
        full_mask.paste(resized_mask, (abs_x0, abs_y0))
        label = (item.get("label") or f"item_{idx}").strip()
        result.append(SegMask(y0=abs_y0, x0=abs_x0, y1=abs_y1, x1=abs_x1, label=label, full_mask_L=full_mask))
    return result
