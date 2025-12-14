import random
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from typing import List, Tuple
from models import SegMask
from utils import parse_segmentation_masks

# 定数定義
MASK_THRESHOLD = 127  # マスクの2値化閾値
FONT_SIZE_MIN = 5
FONT_SIZE_MAX = 30
FONT_SIZE_RATIO = 0.3  # ボックス幅に対するフォントサイズの比率

COLOR_PALETTE = [
    (255,0,0), (0,255,0), (0,0,255),
    (255,255,0), (255,0,255), (0,255,255),
    (255,165,0), (0,128,0), (128,0,128),
    (128,128,0), (0,128,128), (128,0,0),
]

def overlay_mask_on_img(img_rgb: Image.Image, mask_grayscale: Image.Image, color=(255,0,0), alpha=0.5) -> Image.Image:
    """マスクを指定した色とアルファ値で画像に重ねて描画"""
    if img_rgb.mode != "RGBA":
        base = img_rgb.convert("RGBA")
    else:
        base = img_rgb.copy()
    binary_mask = mask_grayscale.point(lambda v: 255 if v > MASK_THRESHOLD else 0).convert("L")
    color_img = Image.new("RGBA", img_rgb.size, color + (0,))
    alpha_channel = (np.array(binary_mask, dtype=np.uint16) * int(alpha*255) // 255).astype(np.uint8)
    overlay = Image.merge("RGBA", (
        Image.new("L", img_rgb.size, color[0]),
        Image.new("L", img_rgb.size, color[1]),
        Image.new("L", img_rgb.size, color[2]),
        Image.fromarray(alpha_channel),
    ))
    return Image.alpha_composite(base, overlay)

def draw_boxes_and_labels(img: Image.Image, masks: List[SegMask]) -> Image.Image:
    """バウンディングボックスとラベルを画像に描画"""
    img = img.convert("RGBA")
    draw = ImageDraw.Draw(img)
    for i, m in enumerate(masks):
        color = tuple(random.randint(0, 255) for _ in range(3))
        draw.rectangle([m.x0, m.y0, m.x1, m.y1], outline=color, width=3)
        text = m.label
        # バウンディングボックスの幅に応じたフォントサイズを計算
        box_width = max(1, m.x1 - m.x0)
        font_size = max(FONT_SIZE_MIN, min(FONT_SIZE_MAX, int(box_width * FONT_SIZE_RATIO)))
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Hiragino Sans GB.ttc", size=font_size)
        except Exception:
            font = ImageFont.load_default()
        tx, ty = m.x0+4, max(12, m.y0-4)
        stroke = 1
        draw.text((tx, ty), text, font=font, fill=color, stroke_fill=(0,0,0), stroke_width=stroke)
    return img

def generate_overlay_image(image_path: str, json_masks: List[dict], mask_alpha=0.5) -> Image.Image:
    """セグメンテーションマスクとラベルを重ねた画像を生成"""
    base = Image.open(image_path).convert("RGB")
    width, height = base.size
    segments = parse_segmentation_masks(json_masks, (width, height))
    composited = base.copy()
    for i, seg in enumerate(segments):
        color = COLOR_PALETTE[i % len(COLOR_PALETTE)]
        composited = overlay_mask_on_img(composited, seg.full_mask_L, color=color, alpha=mask_alpha)
    composited = draw_boxes_and_labels(composited, segments)
    return composited
