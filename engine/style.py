"""House style for headlines: butter-yellow Gveret Levin with a light black halo.

headline() builds a Title in that look, so every reel opens with the same signature title.
"""
import os

from engine.fx import Title

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GVERET = os.path.join(ROOT, "fonts", "GveretLevin.ttf")

BUTTER = (250, 226, 128, 255)       # צהוב חמאה
HALO = (0, 0, 0, 170)               # צל שחור עדין מסביב לאותיות
HALO_BLUR = 7
HALO_DY = 3


def headline(parts, y_base, t_out, size=150, cx=540, color=BUTTER, rise=60):
    """parts = [(text, t_appear)] or [(text, rgba, t_appear)] - one line, right-to-left."""
    parts = [p if len(p) == 3 else (p[0], color, p[1]) for p in parts]
    return Title(parts, GVERET, size, y_base, t_out, cx=cx, weight=None, rise=rise,
                 shadow=HALO, shadow_blur=HALO_BLUR, shadow_dy=HALO_DY)
