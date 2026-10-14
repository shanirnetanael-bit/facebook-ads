"""Hebrew text rendering with Pillow + RAQM (bidi/shaping), never cv2.putText.

render_text() returns an RGBA uint8 patch (text + soft shadow) and the offset of the text box inside it,
cached so a word is rasterised once and then only composited per frame.
"""
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, features

assert features.check("raqm"), "Pillow was built without RAQM - Hebrew would come out reversed"


@lru_cache(maxsize=64)
def get_font(path, size, weight=None):
    f = ImageFont.truetype(path, size, layout_engine=ImageFont.Layout.RAQM)
    if weight is not None:
        try:
            f.set_variation_by_axes([weight])
        except Exception:
            pass
    return f


def text_size(text, font_path, size, weight=None):
    f = get_font(font_path, size, weight)
    l, t, r, b = f.getbbox(text, direction="rtl", language="he")
    return r - l, b - t


@lru_cache(maxsize=2048)
def render_text(text, font_path, size, color, weight=None, shadow=(0, 0, 0, 0), shadow_blur=8, shadow_dy=4, pad=24):
    """color / shadow: RGBA tuples. Returns (rgba uint8 HxWx4, meta) where meta has
    ink_x (left edge of the ink), baseline (y of the text baseline) and ink_w, all in patch pixels,
    so words of different shapes line up on one baseline."""
    f = get_font(font_path, size, weight)
    l, t, r, b = f.getbbox(text, anchor="ls", direction="rtl", language="he")
    w, h = r - l + 2 * pad, b - t + 2 * pad
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).text((pad - l, pad - t), text, font=f, fill=255, anchor="ls", direction="rtl", language="he")
    a = np.asarray(mask).astype(np.float32) / 255.0
    out = np.zeros((h, w, 4), np.float32)
    if shadow[3] > 0:
        sh = np.roll(cv2.GaussianBlur(a, (0, 0), shadow_blur), shadow_dy, axis=0) * (shadow[3] / 255.0)
        out[..., :3] = np.array(shadow[:3], np.float32) / 255.0
        out[..., 3] = sh
    ca = a * (color[3] / 255.0)
    col = np.array(color[:3], np.float32) / 255.0
    # "over" text onto shadow (straight alpha)
    a_out = ca + out[..., 3] * (1 - ca)
    rgb = (col * ca[..., None] + out[..., :3] * (out[..., 3] * (1 - ca))[..., None]) / np.maximum(a_out[..., None], 1e-6)
    res = np.dstack([rgb, a_out])
    return (np.clip(res, 0, 1) * 255).astype(np.uint8), {"ink_x": pad, "baseline": pad - t, "ink_w": r - l}


@lru_cache(maxsize=256)
def rounded_rect(w, h, radius, color, feather=1.5):
    """RGBA patch of an anti-aliased rounded rectangle (supersampled)."""
    s = 4
    img = Image.new("L", (w * s, h * s), 0)
    ImageDraw.Draw(img).rounded_rectangle((0, 0, w * s - 1, h * s - 1), radius=radius * s, fill=255)
    a = np.asarray(img.resize((w, h), Image.LANCZOS)).astype(np.float32) / 255.0
    rgba = np.zeros((h, w, 4), np.float32)
    rgba[..., :3] = np.array(color[:3], np.float32) / 255.0
    rgba[..., 3] = a * color[3] / 255.0
    return (rgba * 255).astype(np.uint8)


def over(frame, rgba, x, y, opacity=1.0, scale=1.0, anchor=(0.5, 0.5)):
    """Composite an RGBA (RGB order) patch onto a BGR uint8 frame in place.
    (x, y) is where `anchor` (fraction of the patch) lands; scale resizes around that anchor."""
    if opacity <= 0.003 or scale <= 0.01:
        return frame
    if abs(scale - 1.0) > 1e-3:
        h0, w0 = rgba.shape[:2]
        rgba = cv2.resize(rgba, (max(1, int(round(w0 * scale))), max(1, int(round(h0 * scale)))),
                          interpolation=cv2.INTER_LINEAR if scale > 1 else cv2.INTER_AREA)
    h, w = rgba.shape[:2]
    x0, y0 = int(round(x - anchor[0] * w)), int(round(y - anchor[1] * h))
    H, W = frame.shape[:2]
    fx0, fy0, fx1, fy1 = max(x0, 0), max(y0, 0), min(x0 + w, W), min(y0 + h, H)
    if fx1 <= fx0 or fy1 <= fy0:
        return frame
    p = rgba[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0].astype(np.float32)
    a = p[..., 3:4] / 255.0 * opacity
    roi = frame[fy0:fy1, fx0:fx1].astype(np.float32)
    roi = roi * (1 - a) + p[..., 2::-1] * a          # RGB -> BGR
    frame[fy0:fy1, fx0:fx1] = np.clip(roi, 0, 255).astype(np.uint8)
    return frame
