"""Reel 02 - "your time is worth a lot of money": entitlement claims, nursing-care (סיעוד) changes
in 2024/2025, success-based fees, call to action.

v1: clean voice, silence cuts with soft transitions, animated captions with gold key words, camera
    moves, motion graphics placed on their words (see words.md), sound design and an original
    rhythmic classical (baroque) music bed ducked under the voice.

Run from the repo root:  python -m projects.reel02.edit
"""
import os
import sys

from engine import sfx
from engine.camera import Camera
from engine.captions import Captions
from engine.fx import (Behind, Checklist, Clock, DownArrows, EndCard, NameCard, NumberFlip, ShineCard, Stamp,
                       Timeline, Title)
from engine.project import Project
from engine.render import contact_sheet, render
from engine.timemap import Clip

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SOURCE = os.path.join(ROOT, "input", "reel02.mp4")
WORK = os.path.join(ROOT, "work", "reel02")
OUT = os.path.join(ROOT, "out")

NAVY = (11, 27, 52, 255)
GOLD = (212, 175, 55, 255)
WHITE = (255, 255, 255, 255)
FONT = os.path.join(ROOT, "fonts", "Heebo.ttf")

# On-screen spelling fixes (word index -> text); timing stays with the spoken word.
# Words marked "?" were unclear in the recording - confirm with the speaker.
CORRECTIONS = {
    18: "במיצוי",
    21: "בתביעות", 22: "מול", 24: "הביטוח",
    25: "בתחום", 26: "הסיעוד.",              # ? recording unclear here
    34: "הלבשה,",
    40: "ייתכן", 44: "שנאמד",
    50: "הסתכלתי", 52: "השעון,",
    55: "ב-2024", 56: "ו-2025", 57: "וצפויים",
    69: "ככה", 71: "להוכיח", 76: "להוכיח",
    83: "הפוליסה", 84: "השתנתה.",
    88: "להתעכב.",                             # ?
    94: "תביעה,", 97: "נדחתם,",
    99: "מבקש", 100: "מכם", 102: "סבלנות,",    # ?
    113: "עובדים", 124: "ואתן", 127: "חוות",
    137: "זאת",                                # ?
    141: "שנותר", 144: "להשאיר", 145: "פרטים",
    150: "שכבר", 151: "קיבלו", 159: "נדחו.",
}

# Key words for the average viewer: money, the law changes, rejection, what to do -> gold + underline
EMPHASIS = {1, 6, 14, 15, 18, 19, 21, 43, 45, 46, 47, 52, 55, 56, 67, 68, 73, 78, 84, 88, 97, 107, 115, 118,
            121, 127, 128, 129, 146, 148, 149, 154, 159}

CAPTION_STYLE = {
    "font": FONT, "size": 70, "weight": 800,
    "color": WHITE, "accent": GOLD, "shadow": (0, 0, 0, 0),
    "pill": (NAVY[0], NAVY[1], NAVY[2], 225), "pad_x": 34, "pill_h": 112, "radius": 30,
    "y": 1390, "baseline_off": 26,
    "space": 20, "max_words": 3, "max_width": 800, "pause_break": 0.30,
    "lead": 0.05, "hold": 0.35, "out_dur": 0.12, "pill_in": 0.18,
    "word_lead": 0.05, "word_fade": 0.10, "word_pop": 0.22, "rise": 16,
    "grow_lead": 0.07, "grow": 0.12, "min_show": 0.35,
}

CHECKLIST_WORDS = range(27, 40)       # the checklist shows these words itself -> no captions under it


def build():
    p = Project(SOURCE, WORK, CORRECTIONS, clean_voice=True)
    tm = p.cut_silences(keep_gap=0.14, pad_before=0.06, pad_after=0.08, xfade=0.10)
    # the opening breath + "חברים" is one gesture: keep it in one piece
    tm.segs[0:2] = [Clip(tm.segs[0].src_start, tm.segs[1].src_end)]
    tm._relayout()
    w = p.w
    t_end = tm.duration
    END = 3.4
    tm.add_freeze(tm.segs[-1].src_end - 0.04, END)

    # --- camera ------------------------------------------------------------------------------
    cam = Camera(p.W, p.H, focus=(540, 560))
    cam.intro = (0.6, 1.15)                                     # hook: quick pull-back on the walk-in
    looks = [1.0, 1.07, 1.0, 1.05, None, 1.0, 1.08, 1.0, 1.06, 1.0, 1.07, None, 1.0, 1.06]
    for k, seg in enumerate(tm.segs[:-1]):
        z = looks[k % len(looks)]
        if z is None:                                            # slow push: builds tension on long takes
            cam.set_base(seg.out_start, seg.out_start + seg.dur, 1.0, 1.10)
        else:
            cam.set_base(seg.out_start, seg.out_start + seg.dur, z)
    cam.set_base(t_end, t_end + END, 1.0, 1.04)
    cam.punch(w(6), 0.10, attack=0.07, hold=0.45)                # "כסף"
    cam.punch(w(14), 0.05, hold=0.5)                             # "עורך דין"
    cam.punch(w(43), 0.05, hold=0.3, release=0.2)                # "פיצוי"
    cam.punch(w(45), 0.08, attack=0.08, hold=0.9)                # "מאות אלפי שקלים"
    cam.shake(w(67), 0.28, 12)                                   # "לרעת"
    cam.punch(w(78), 0.08, attack=0.07, hold=0.5)                # "ארבע"
    cam.punch(w(84), 0.05, hold=0.35)                            # "השתנתה"
    cam.shake(w(97), 0.30, 15)                                   # "נדחתם" (stamp impact)
    cam.punch(w(118), 0.07, hold=0.5)                            # "הצלחה"
    cam.punch(w(121), 0.04, hold=0.4)                            # "אישי"
    cam.punch(w(159), 0.06, hold=0.4)                            # "נדחו"
    p.camera = cam

    # --- graphics (each on its word) ---------------------------------------------------------
    navy, gold = NAVY, GOLD
    p.layers += [
        # hook
        Behind(Title([("הזמן", WHITE, w(1)), ("שלכם", WHITE, w(2))], FONT, 125, 290, t_out=w(7) - 0.1,
                     stroke=4, stroke_color=navy)),
        Behind(Title([("שווה", WHITE, w(3)), ("הרבה", WHITE, w(4)), ("כסף", gold, w(6))], FONT, 125, 425,
                     t_out=w(7) - 0.1, stroke=4, stroke_color=navy)),
        NameCard("שניר נתנאל", "עורך דין · מיצוי זכויות", FONT, t_in=w(9) - 0.1, t_out=w(17)),
        Behind(Title([("מיצוי", WHITE, w(18)), ("זכויות", gold, w(19))], FONT, 130, 420, t_out=w(21) - 0.1,
                     stroke=4, stroke_color=navy)),
        Behind(Title([("מול", WHITE, w(22)), ("חברות", WHITE, w(23)), ("הביטוח", WHITE, w(24))], FONT, 120, 420,
                     t_out=w(27) - 0.1, stroke=4, stroke_color=navy)),
        # the daily activities (ADL) - checklist under the speaker
        Checklist("קושי באחת הפעולות?", [("הלבשה", w(34)), ("רחצה", w(35)), ("הליכה", w(36)),
                                          ("שליטה בסוגרים", w(37))],
                  FONT, t_header=w(27), t_out=w(40) - 0.1, chip_w=440, item_size=50),
        Behind(ShineCard([("מאות", w(45)), ("אלפי", w(46)), ("₪", w(47))], FONT, 120, cy=280,
                         t_out=w(48) - 0.05, t_shine=w(47) + 0.15)),
        # "I didn't look at the clock for nothing"
        Clock(t_in=w(50) - 0.05, t_out=w(53) - 0.1, x=830, y=330),
        Behind(Timeline([("2024", w(55)), ("2025", w(56)), ("?", w(57))], FONT, t_out=w(61) - 0.1, y=260)),
        Behind(Title([("לרעת", gold, w(67)), ("המבוטחים", WHITE, w(68))], FONT, 120, 420, t_out=w(69) - 0.1,
                     stroke=4, stroke_color=navy)),
        Behind(NumberFlip(3, 4, "פעולות", FONT, t_in=w(73) - 0.05, t_flip=w(78), t_out=w(80) - 0.05, cy=300)),
        Behind(Title([("הפוליסה", WHITE, w(83)), ("השתנתה", gold, w(84))], FONT, 120, 420, t_out=w(85) - 0.05,
                     stroke=4, stroke_color=navy)),
        Clock(t_in=w(86) - 0.05, t_out=w(89) - 0.1, x=830, y=330, turns=2.5),
        Stamp("נדחה", FONT, t_in=w(97), t_out=w(98) + 0.6, x=250, y=560),
        DownArrows(t_in=w(104), t_out=w(108)),
        Behind(ShineCard([("רק", w(115)), ("על בסיס", w(116)), ("הצלחה", w(118))], FONT, 110, cy=280,
                         t_out=w(119) + 0.4, t_shine=w(118) + 0.15)),
        Behind(Title([("חוות", WHITE, w(127)), ("דעת", WHITE, w(128)), ("אישית", gold, w(129))], FONT, 125, 420,
                     t_out=w(134) - 0.05, stroke=4, stroke_color=navy)),
        DownArrows(t_in=w(144), t_out=w(147) + 0.3),
        Behind(Title([("מאות", gold, w(148)), ("לקוחות", WHITE, w(149))], FONT, 125, 420, t_out=w(152),
                     stroke=4, stroke_color=navy)),
    ]
    caption_words = [x for x in p.words_out() if x["i"] not in CHECKLIST_WORDS]
    caps = Captions(caption_words, dict(CAPTION_STYLE, emph=EMPHASIS), p.W, p.H)
    p.layers.append(lambda frame, t, ctx, info: caps.draw(frame, t))
    p.layers.append(EndCard("שניר נתנאל", "עורך דין · מיצוי זכויות", FONT, t_in=t_end,
                            cta=("השאירו פרטים למטה", "לחוות דעת אישית")))

    # --- sound design (dB, all well under the voice) --------------------------------------------
    wh = sfx.whoosh()
    soft = sfx.whoosh(0.35, 600, 3500)
    coin = sfx.pop(1900, 1500, 0.07)
    p.sfx += [
        (0.0, sfx.whoosh(0.5, 400, 5000), -17),                          # walk-in
        (w(1) - 0.12, soft, -24),
        (w(6) - 0.02, coin, -17), (w(6) + 0.06, sfx.shimmer(0.9), -20),    # "כסף" cha-ching
        (w(9) - 0.15, soft, -24),                                         # name card
        (w(18) - 0.12, wh, -22), (w(22) - 0.12, soft, -24),
        (w(27) - 0.05, sfx.pop(), -22),                                    # checklist header
        *[(w(i) + 0.08, sfx.tick(), -18) for i in (34, 35, 36, 37)],       # each check lands
        (w(45) - 0.12, wh, -21), (w(47) + 0.15, sfx.shimmer(), -20),       # hundreds of thousands
        *[(w(50) + k * 0.16, sfx.tick(), -21) for k in range(7)],          # clock ticking
        *[(w(i), sfx.pop(1000, 600, 0.08), -20) for i in (55, 56)],        # 2024, 2025
        (w(57), sfx.pop(600, 380, 0.1), -20),                              # "?"
        (w(67) - 0.01, sfx.thud(0.35), -13),                               # "לרעת"
        (w(73) - 0.05, sfx.pop(), -21),
        (w(78) - 0.25, sfx.whoosh(0.25, 900, 4000), -24),                 # cross-out
        (w(78) + 0.08, sfx.thud(0.3), -15),                                # 4 lands
        (w(83) - 0.12, soft, -24),
        *[(w(86) + k * 0.12, sfx.tick(), -22) for k in range(6)],          # clock again, faster
        (w(97) - 0.01, sfx.thud(), -9),                                    # stamp
        (w(104) - 0.1, sfx.whoosh(0.4, 300, 2500), -22),                   # arrows down
        (w(115) - 0.12, wh, -22), (w(118) + 0.15, sfx.shimmer(), -20),
        (w(127) - 0.12, soft, -23),
        (w(144) - 0.1, sfx.whoosh(0.4, 300, 2500), -22),
        (w(148) - 0.12, soft, -23),
        (t_end, sfx.whoosh(0.6, 300, 3000), -20),
        (t_end + 0.3, sfx.shimmer(1.4), -19),
        (t_end + 1.15, sfx.pop(800, 500, 0.09), -22),                     # CTA button lands
    ]
    p.music = dict(style="classical", bpm=112, under_db=-24.0, open_db=-15.0)
    p.prepare_audio()
    return p


if __name__ == "__main__":
    p = build()
    print(p.tm.describe())
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, sys.argv[1] if len(sys.argv) > 1 else "reel02_v1.mp4")
    render(p, out)
    print("rendered", out)
    for page in contact_sheet(out, os.path.join(WORK, "contact_v1")):
        print("contact", page)
