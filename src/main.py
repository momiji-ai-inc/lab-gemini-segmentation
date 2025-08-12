import os
import argparse
from dotenv import load_dotenv
from segment import call_gemini
from overlay import generate_overlay_image

load_dotenv()
os.makedirs("outputs", exist_ok=True)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash-lite"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--query", default="all clearly visible objects")
    ap.add_argument("--alpha", type=float, default=0.5)
    args = ap.parse_args()
    print("1. calling gemini api")
    masks = call_gemini(args.image, args.query, GEMINI_MODEL, GEMINI_API_KEY)
    print("2. generating overlay image")
    out_img = generate_overlay_image(args.image, masks, mask_alpha=args.alpha)
    out_img.save("outputs/overlay.png")

if __name__ == "__main__":
    main()
