"""Reel 01 - nursing-care insurance (פוליסת סיעוד). Version 1: silence cuts + animated captions.

Run from the repo root:  python -m projects.reel01.edit
"""
import os
import sys

from engine.captions import Captions
from engine.project import Project
from engine.render import contact_sheet, render

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SOURCE = os.path.join(ROOT, "input", "source.mp4")
WORK = os.path.join(ROOT, "work")
OUT = os.path.join(ROOT, "out")

# Brand palette (RGBA): navy / gold / white - clean & minimal
NAVY = (11, 27, 52, 255)
GOLD = (212, 175, 55, 255)
WHITE = (255, 255, 255, 255)
FONT = os.path.join(ROOT, "fonts", "Heebo.ttf")

# Transcript fixes for on-screen text (word index -> text). Timing stays with the original word.
CORRECTIONS = {
    7: "תביעות",
    16: "עורך דין,",
    19: "תביעות",
    35: "סיעוד.",
    73: "שנאמד",
    83: "הלבשה,",
}

CAPTION_STYLE = {
    "font": FONT, "size": 70, "weight": 800,
    "color": WHITE, "accent": GOLD, "shadow": (0, 0, 0, 0),
    "pill": (NAVY[0], NAVY[1], NAVY[2], 225), "pad_x": 34, "pill_h": 112, "radius": 30,
    "y": 1390, "baseline_off": 26,             # pill centre y; baseline offset from it
    "space": 20, "max_words": 3, "max_width": 840, "pause_break": 0.30,
    "lead": 0.05, "hold": 0.35, "out_dur": 0.12, "pill_in": 0.18,
    "word_lead": 0.05, "word_fade": 0.10, "word_pop": 0.22, "rise": 16,
    "grow_lead": 0.07, "grow": 0.12, "min_show": 0.35,
}


def build():
    p = Project(SOURCE, WORK, CORRECTIONS)
    p.cut_silences(keep_gap=0.14, pad_before=0.06, pad_after=0.08, xfade=0.10)
    caps = Captions(p.words_out(), CAPTION_STYLE, p.W, p.H)
    p.layers.append(lambda frame, t, ctx, info: caps.draw(frame, t))
    p.prepare_audio()
    return p


if __name__ == "__main__":
    p = build()
    print(p.tm.describe())
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, sys.argv[1] if len(sys.argv) > 1 else "reel01_v1.mp4")
    render(p, out)
    print("rendered", out)
    for page in contact_sheet(out, os.path.join(WORK, "contact_v1")):
        print("contact", page)
