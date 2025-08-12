import json
from google import genai
from google.genai import types
from PIL import Image
from utils import parse_segmentation_masks

PROMPT_TEMPLATE = """Give the segmentation masks for {desc} in the image.
Output a JSON list where each item has:
- "box_2d": [y0, x0, y1, x1] normalized to 0-1000
- "mask": base64 PNG (probability map 0-255) cropped to the box
- "label": a descriptive text label
Use descriptive labels."""

def call_gemini(image_path: str, desc: str, model: str, api_key: str) -> list:
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
    text = resp.text or ""
    if "```" in text:
        text = text.split("```", 1)[1]
        text = text.split("```", 1)[0]
        if text.lstrip().startswith("json"):
            text = text.split("\n", 1)[1] if "\n" in text else ""
    data = json.loads(text)
    
    # レスポンス結果をjsonとして保存
    with open(f"outputs/response.json", "w", encoding="utf-8") as f:
        json.dump(masks, f, ensure_ascii=False, indent=2)
    return data if isinstance(data, list) else []
