import argparse
import json
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from typing import List, Tuple
from models import SegMask
from utils import parse_segmentation_masks

def overlay_mask_on_img(img_rgb: Image.Image, mask_L: Image.Image, color=(255,0,0), alpha=0.5) -> Image.Image:
    if img_rgb.mode != "RGBA":
        base = img_rgb.convert("RGBA")
    else:
        base = img_rgb.copy()
    bin_mask = mask_L.point(lambda v: 255 if v > 127 else 0).convert("L")
    color_img = Image.new("RGBA", img_rgb.size, color + (0,))
    a = (np.array(bin_mask, dtype=np.uint16) * int(alpha*255) // 255).astype(np.uint8)
    overlay = Image.merge("RGBA", (
        Image.new("L", img_rgb.size, color[0]),
        Image.new("L", img_rgb.size, color[1]),
        Image.new("L", img_rgb.size, color[2]),
        Image.fromarray(a),
    ))
    return Image.alpha_composite(base, overlay)

def draw_boxes_and_labels(img: Image.Image, masks: List[SegMask]) -> Image.Image:
    img = img.convert("RGBA")
    draw = ImageDraw.Draw(img)
    colors = [
        (255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),
        (255,165,0),(0,128,0),(128,0,128),(128,128,0),(0,128,128),(128,0,0),
    ]
    for i, m in enumerate(masks):
        color = colors[i % len(colors)]
        draw.rectangle([m.x0, m.y0, m.x1, m.y1], outline=color, width=3)
        text = m.label
        # バウンディングボックスの横幅の30%をフォントサイズに（下限6, 上限40）
        box_width = max(1, m.x1 - m.x0)
        font_size = max(6, min(40, int(box_width * 0.3)))
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Hiragino Sans GB.ttc", size=font_size)
        except Exception:
            font = ImageFont.load_default()
        tx, ty = m.x0+4, max(12, m.y0-4)
        stroke = 1
        draw.text((tx, ty), text, font=font, fill=color, stroke_fill=(0,0,0), stroke_width=stroke)
    return img

def generate_overlay_image(image_path: str, json_masks: List[dict], mask_alpha=0.5) -> Image.Image:
    base = Image.open(image_path).convert("RGB")
    W, H = base.size
    segs = parse_segmentation_masks(json_masks, (W, H))
    composited = base.copy()
    palette = [
        (255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),
        (255,165,0),(0,128,0),(128,0,128),(128,128,0),(0,128,128),(128,0,0),
    ]
    for i, s in enumerate(segs):
        color = palette[i % len(palette)]
        composited = overlay_mask_on_img(composited, s.full_mask_L, color=color, alpha=mask_alpha)
    composited = draw_boxes_and_labels(composited, segs)
    return composited
