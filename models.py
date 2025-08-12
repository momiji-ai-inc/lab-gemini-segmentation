from pydantic import BaseModel, Field
from typing import Any
from PIL import Image

class SegMask(BaseModel):
    y0: int
    x0: int
    y1: int
    x1: int
    label: str
    full_mask_L: Any = Field(..., description="画像と同サイズ（L, 0-255）")

    class Config:
        arbitrary_types_allowed = True
