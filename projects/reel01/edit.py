"""Reel 01 - nursing-care insurance (פוליסת סיעוד).
v1: silence cuts + animated captions.
v3: + original background music (ducked under the voice) + call to action on the end card.
v2: + transcript fixes, camera moves, headlines behind the person, name card, stamp, money card,
    checklist, end card and sound design - every effect anchored to its word number (see words.md).

Run from the repo root:  python -m projects.reel01.edit
"""
import os
import sys

from engine import sfx
from engine.camera import Camera
from engine.captions import Captions
from engine.fx import Behind, Checklist, EndCard, NameCard, ShineCard, Stamp, Title
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
    46: "נדחים,",
    50: "יום",
    63: "סובל",
    73: "שנאמד",
    83: "הלבשה,",
}

# Key words that stay gold with an underline in the captions
EMPHASIS = {7, 8, 32, 35, 46, 55, 72, 76, 77, 78}

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


def build(version=3):
    p = Project(SOURCE, WORK, CORRECTIONS)
    p.cut_silences(keep_gap=0.14, pad_before=0.06, pad_after=0.08, xfade=0.10)
    if version < 2:
        caps = Captions(p.words_out(), CAPTION_STYLE, p.W, p.H)
        p.layers.append(lambda frame, t, ctx, info: caps.draw(frame, t))
        p.prepare_audio()
        return p

    w = p.w                                   # w(i) = output time of word i
    speech_end = p.tm.duration
    END = 3.6 if version >= 3 else 2.4
    p.tm.add_freeze(p.tm.segs[-1].src_end - 0.04, END)          # outro hold on the last frame
    t_end = speech_end

    # --- camera ------------------------------------------------------------------------------
    cam = Camera(p.W, p.H, focus=(620, 560))
    cam.intro = (0.7, 1.18)                                      # hook: fast pull-back on frame 1
    zooms = [1.0, 1.07, 1.0, None, 1.06, 1.0]                     # alternate framing per clip (jump-cut energy)
    for k, seg in enumerate(p.tm.segs[:-1]):
        a, b = seg.out_start, seg.out_start + seg.dur
        if zooms[k] is None:                                     # clip 4 (words 57-78): slow push builds tension
            cam.set_base(a, b, 1.0, 1.12)
        else:
            cam.set_base(a, b, zooms[k])
    cam.set_base(t_end, t_end + END, 1.0, 1.04)                   # gentle drift on the outro
    cam.punch(w(7), 0.06, hold=1.0)                               # "תביעות ביטוח"
    cam.punch(w(30), 0.05, hold=0.25, release=0.2)                # "הרבה"
    cam.punch(w(32), 0.10, attack=0.08, hold=0.35)                # "כסף"
    cam.punch(w(42), 0.04, hold=0.3, release=0.1)                 # "פעם"
    cam.punch(w(44), 0.09, attack=0.08, hold=0.5)                 # "פעמיים"
    cam.shake(w(46), 0.22, 8)                                     # "נדחים"
    cam.shake(w(55), 0.32, 16)                                    # "דוחה" (stamp impact)
    cam.punch(w(67), 0.05, hold=0.4)                              # "ייתכן"
    cam.punch(w(76), 0.08, hold=1.0)                              # "מאות אלפי שקלים"
    p.camera = cam

    # --- graphics ----------------------------------------------------------------------------
    navy, gold = NAVY, GOLD
    p.layers += [
        Behind(Title([("תביעות", navy, w(7)), ("ביטוח", navy, w(8))], FONT, 140, 415, t_out=w(9) - 0.05)),
        NameCard("שניר נתנאל", "עורך דין · תביעות ביטוח", FONT, t_in=w(13) - 0.1, t_out=w(21) - 0.3),
        Behind(Title([("פוליסת", navy, w(34)), ("סיעוד", gold, w(35))], FONT, 132, 415, t_out=w(36) - 0.15,
                     stroke=4, stroke_color=navy)),
        Stamp("נדחה", FONT, t_in=w(55), t_out=w(57) - 0.1, x=245, y=560),
        Behind(ShineCard([("מאות", w(76)), ("אלפי", w(77)), ("₪", w(78))], FONT, 120, cy=250,
                         t_out=w(79) - 0.1, t_shine=w(78) + 0.15)),
        Checklist("הפעולות:", [("הליכה", w(81)), ("רחצה", w(82)), ("הלבשה", w(83)), ("ניידות", w(84))],
                  FONT, t_header=w(79) - 0.05, t_out=t_end),
    ]
    caption_words = [x for x in p.words_out() if x["i"] < 79]   # the checklist takes over from word 79
    caps = Captions(caption_words, dict(CAPTION_STYLE, emph=EMPHASIS), p.W, p.H)
    p.layers.append(lambda frame, t, ctx, info: caps.draw(frame, t))
    cta = ("מוזמנים ליצור קשר", "לשיחת ייעוץ ללא עלות") if version >= 3 else None
    p.layers.append(EndCard("שניר נתנאל", "עורך דין · תביעות ביטוח", FONT, t_in=t_end, cta=cta))

    # --- sound design (levels in dB, under the voice) -------------------------------------------
    whoosh = sfx.whoosh()
    p.sfx += [
        (0.0, sfx.whoosh(0.5, 400, 5000), -16),
        (w(7) - 0.12, whoosh, -20),
        (w(13) - 0.15, sfx.whoosh(0.35, 600, 3500), -24),
        (w(34) - 0.12, whoosh, -20),
        (w(32), sfx.pop(700, 350, 0.1), -18),
        (w(67), sfx.whoosh(0.3, 800, 3000), -26),
        (w(55) - 0.01, sfx.thud(), -9),
        (w(76) - 0.1, sfx.whoosh(0.4, 500, 4000), -22),
        (w(78) + 0.15, sfx.shimmer(), -20),
        (w(79) - 0.05, sfx.pop(), -22),
        *[(w(i) + 0.08, sfx.tick(), -18) for i in (81, 82, 83, 84)],
        (t_end, sfx.whoosh(0.6, 300, 3000), -20),
        (t_end + 0.3, sfx.shimmer(1.4), -19),
    ]
    if version >= 3:
        p.sfx.append((t_end + 1.15, sfx.pop(800, 500, 0.09), -22))   # CTA button lands
        p.music = dict(bpm=84, under_db=-23.0, open_db=-14.0)
    p.prepare_audio()
    return p


if __name__ == "__main__":
    version = int(os.environ.get("VERSION", "3"))
    p = build(version)
    print(p.tm.describe())
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, sys.argv[1] if len(sys.argv) > 1 else f"reel01_v{version}.mp4")
    render(p, out)
    print("rendered", out)
    for page in contact_sheet(out, os.path.join(WORK, f"contact_v{version}")):
        print("contact", page)
