import os, io, json, base64
from dataclasses import dataclass
from typing import List, Tuple
from PIL import Image, ImageDraw
import numpy as np

# Google GenAI SDK
from google import genai
from google.genai import types
from dotenv import load_dotenv
load_dotenv()

# ==== 設定 ====
MODEL = "gemini-2.5-flash"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("GEMINI_API_KEY environment variable is not set. Please set it in .env or your environment.")
    raise SystemExit("GEMINI_API_KEY not set")
client = genai.Client(api_key=GEMINI_API_KEY)

PROMPT_TEMPLATE = """
Give the segmentation masks for the objects described below.
Output a JSON list where each item has:
- "box_2d": [y0, x0, y1, x1] normalized to 0–1000
- "mask": base64 PNG (probability map 0–255) cropped to the box
- "label": a descriptive text label

Query: "{query}"
"""

@dataclass
class SegItem:
    label: str
    box: Tuple[int, int, int, int]  # (y0, x0, y1, x1) absolute px
    mask_img: Image.Image           # PIL Image (L) resized to box size

def _parse_codefence_json(s: str) -> str:
    """Docsやブログのコードフェンス付きJSONを想定して除去。"""
    if "```json" in s:
        s = s.split("```json", 1)[1]
        s = s.split("```", 1)[0]
    return s

def segment(image_path: str, query: str, output_dir: str = "outputs") -> List[SegItem]:
    # 画像を読み込み＆長辺1024にサムネイル（推奨）
    im = Image.open(image_path).convert("RGB")
    im.thumbnail([1024, 1024], Image.Resampling.LANCZOS)
    W, H = im.size

    # thinking を無効化
    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        response_mime_type="application/json"
    )

    prompt = PROMPT_TEMPLATE.format(query=query).strip()

    # SDKはPillow画像をそのまま渡せる
    resp = client.models.generate_content(
        model=MODEL,
        contents=[prompt, im],
        config=config,
    )

    # JSON取り出し
    raw = _parse_codefence_json(resp.text)
    items = json.loads(raw)

    os.makedirs(output_dir, exist_ok=True)
    results: List[SegItem] = []

    # 各マスクを復元して重ね合わせ画像も保存
    overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for i, it in enumerate(items):
        label = it.get("label", f"item_{i}")
        y0, x0, y1, x1 = it["box_2d"]
        # 正規化(0–1000)→絶対px
        ay0 = int(y0 / 1000 * H); ax0 = int(x0 / 1000 * W)
        ay1 = int(y1 / 1000 * H); ax1 = int(x1 / 1000 * W)
        if ay0 >= ay1 or ax0 >= ax1:  # 異常ボックスはスキップ
            continue

        # マスクPNG（"data:image/png;base64,..."）
        b64 = it["mask"]
        if b64.startswith("data:image/png;base64,"):
            b64 = b64.split(",", 1)[1]
        mask = Image.open(io.BytesIO(base64.b64decode(b64))).convert("L")
        # 箱サイズにリサイズ
        mask = mask.resize((ax1 - ax0, ay1 - ay0), Image.Resampling.BILINEAR)

        # 合成（127閾値で2値化）
        arr = np.array(mask)
        yy, xx = np.where(arr > 127)
        for (yyi, xxi) in zip(yy, xx):
            draw.point((ax0 + xxi, ay0 + yyi), fill=(255, 0, 0, 100))

        # 個別保存
        mask.save(os.path.join(output_dir, f"{i:02d}_{label}_mask.png"))
        results.append(SegItem(label=label, box=(ay0, ax0, ay1, ax1), mask_img=mask))

    # 元画像にオーバーレイを重ねた合成画像
    composite = Image.alpha_composite(im.convert("RGBA"), overlay)
    out_path = os.path.join(output_dir, "overlay.png")
    composite.save(out_path)
    print(f"Saved overlay to {out_path}")

    # 検出結果の簡易ログ
    for r in results:
        print(f"- {r.label}: box(y0,x0,y1,x1)={r.box}")

    return results

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="Path to input image")
    p.add_argument("--query", required=True, help='e.g. "the people who are not sitting"')
    p.add_argument("--out", default="outputs")
    args = p.parse_args()

    segment(args.image, args.query, args.out)
