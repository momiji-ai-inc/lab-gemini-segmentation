import os
import json
import argparse
from dotenv import load_dotenv
from segment import call_gemini
from overlay import generate_overlay_image
from datetime import datetime

load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash-lite"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--query", default="all clearly visible objects")
    ap.add_argument("--out", default="outputs/overlay.png")
    ap.add_argument("--alpha", type=float, default=0.5)
    args = ap.parse_args()

    print("== start: call_gemini")
    masks = call_gemini(args.image, args.query, GEMINI_MODEL, GEMINI_API_KEY)
    print(f"== end: call_gemini, start: save json")

    os.makedirs("outputs", exist_ok=True)
    json_path = f"outputs/{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(masks, f, ensure_ascii=False, indent=2)
    print(f"Saved API result: {json_path}")

    print("== start: generate_overlay_image")
    out_img = generate_overlay_image(args.image, masks, mask_alpha=args.alpha)
    print(f"== end: generate_overlay_image, start: save")
    out_img.save(args.out)
    print("== end: all")

if __name__ == "__main__":
    main()
