import os, io, json, base64, argparse
from dotenv import load_dotenv
from typing import List, Tuple
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from models import SegMask
from google import genai
from google.genai import types

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
DEFAULT_MODEL = "gemini-2.5-flash"
PROMPT_TEMPLATE = """Give the segmentation masks for {desc} in the image.
Output a JSON list where each item has:
- "box_2d": [y0, x0, y1, x1] normalized to 0-1000
- "mask": base64 PNG (probability map 0-255) cropped to the box
- "label": a descriptive text label
Use descriptive labels."""

def _as_abs_box(box_2d, W, H) -> Tuple[int, int, int, int]:
    y0, x0, y1, x1 = box_2d
    ay0 = int(np.floor(y0 / 1000.0 * H))
    ax0 = int(np.floor(x0 / 1000.0 * W))
    ay1 = int(np.ceil (y1 / 1000.0 * H))
    ax1 = int(np.ceil (x1 / 1000.0 * W))
    return ay0, ax0, ay1, ax1

def _decode_mask_to_L(mask_b64_png: str) -> Image.Image:
    if mask_b64_png.startswith("data:image/png;base64,"):
        mask_b64_png = mask_b64_png.split(",", 1)[1]
    buf = io.BytesIO(base64.b64decode(mask_b64_png))
    return Image.open(buf).convert("L")

def parse_segmentation_masks(items: List[dict], img_size: Tuple[int,int]) -> List[SegMask]:
    W, H = img_size
    out: List[SegMask] = []
    for idx, it in enumerate(items):
        if not isinstance(it, dict) or "box_2d" not in it or "mask" not in it:
            continue
        ay0, ax0, ay1, ax1 = _as_abs_box(it["box_2d"], W, H)
        if not (0 <= ax0 < ax1 <= W and 0 <= ay0 < ay1 <= H):
            continue

        # 箱サイズにマスクをリサイズ → フルサイズキャンバスへ合成
        m = _decode_mask_to_L(it["mask"]).resize((ax1-ax0, ay1-ay0), Image.Resampling.BICUBIC)
        full = Image.new("L", (W, H), color=0)
        full.paste(m, (ax0, ay0))
        label = (it.get("label") or f"item_{idx}").strip()
        out.append(SegMask(ay0, ax0, ay1, ax1, label, full))
    return out

def overlay_mask_on_img(img_rgb: Image.Image, mask_L: Image.Image, color=(255,0,0), alpha=0.5) -> Image.Image:
    """mask_L>127 の画素へ color を alpha で重畳"""
    if img_rgb.mode != "RGBA":
        base = img_rgb.convert("RGBA")
    else:
        base = img_rgb.copy()

    # 2値化してアルファに使う（0..255）
    bin_mask = mask_L.point(lambda v: 255 if v > 127 else 0).convert("L")
    color_img = Image.new("RGBA", img_rgb.size, color + (0,))
    # 指定 alpha を掛けたアルファチャンネル
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
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=max(16, img.height//25))
    except:
        font = ImageFont.load_default()

    for i, m in enumerate(masks):
        color = colors[i % len(colors)]
        draw.rectangle([m.x0, m.y0, m.x1, m.y1], outline=color, width=3)
        # ラベル（白縁取り）
        text = m.label
        tx, ty = m.x0+4, max(12, m.y0-4)
        stroke = 1
        draw.text((tx, ty), text, font=font, fill=color, stroke_fill=(0,0,0), stroke_width=stroke)
    return img

def generate_overlay_image(image_path: str, json_masks: List[dict], mask_alpha=0.5) -> Image.Image:
    # 入力画像
    base = Image.open(image_path).convert("RGB")
    W, H = base.size
    segs = parse_segmentation_masks(json_masks, (W, H))
    # 先にすべてのマスクを重畳
    composited = base.copy()
    palette = [
        (255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),
        (255,165,0),(0,128,0),(128,0,128),(128,128,0),(0,128,128),(128,0,0),
    ]
    for i, s in enumerate(segs):
        color = palette[i % len(palette)]
        composited = overlay_mask_on_img(composited, s.full_mask_L, color=color, alpha=mask_alpha)
    # 枠とラベルを描画
    composited = draw_boxes_and_labels(composited, segs)
    return composited

def call_gemini(image_path: str, desc: str, model: str, api_key: str) -> List[dict]:
    client = genai.Client(api_key=api_key)
    im = Image.open(image_path).convert("RGB")

    cfg = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        response_mime_type="application/json",
    )
    prompt = PROMPT_TEMPLATE.format(desc=(desc or "all clearly visible objects"))
    resp = client.models.generate_content(
        model=model,
        contents=[im, prompt],
        config=cfg,
    )
    # レスポンスはJSON文字列（``json フェンス対応）
    text = resp.text or ""
    if "```" in text:
        # ```json ... ``` の場合
        text = text.split("```", 1)[1]
        text = text.split("```", 1)[0]
        if text.lstrip().startswith("json"):
            text = text.split("\n", 1)[1] if "\n" in text else ""
    return json.loads(text)

def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--query", default="all clearly visible objects")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default="overlay.png")
    ap.add_argument("--alpha", type=float, default=0.5)
    args = ap.parse_args()

    print("== start: call_gemini")
    masks = call_gemini(args.image, args.query, args.model, GEMINI_API_KEY)
    from datetime import datetime
    print("== end: call_gemini, start: save json")
    os.makedirs("outputs", exist_ok=True)
    json_path = f"outputs/{datetime.now().strftime('%Y%m%d')}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(masks, f, ensure_ascii=False, indent=2)
    print(f"Saved API result: {json_path}")
    print("== start: generate_overlay_image")
    out_img = generate_overlay_image(args.image, masks, mask_alpha=args.alpha)
    print("== end: generate_overlay_image, start: save")
    out_img.save(args.out)
    print(f"Saved: {args.out}")
    print("== end: all")

if __name__ == "__main__":
    main()
