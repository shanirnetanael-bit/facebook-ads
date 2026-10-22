"""Siri reel (comedy) - driving, she says the name deliberately mangled ("ס-שה") so Siri will dial the right
person; Siri repeats it, the call rings, she laughs to camera.
The source already has burned-in captions, so this edit adds framing, a hook headline, pacing and sound.

Layout: the 1080x1500 source sits as a rounded card on a blurred copy of itself inside a 1080x1920 frame;
the top band carries the hook headline in the house style (Gveret Levin, butter yellow).

Run from the repo root:  python -m projects.siri.edit <source.mov>
"""
import os
import sys

import cv2
import numpy as np

from engine import sfx
from engine.audio import build_audio, load_audio, loudnorm_file, mix_sfx, write_wav
from engine.camera import Camera
from engine.render import SourceReader, blend_sources, contact_sheet, render
from engine.style import headline
from engine.text import rounded_rect
from engine.timemap import TimeMap

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORK = os.path.join(ROOT, "work", "siri")
OUT = os.path.join(ROOT, "out")

SRC_W, SRC_H = 1080, 1500
CARD_SCALE = 0.95
CARD_W, CARD_H = int(SRC_W * CARD_SCALE), int(SRC_H * CARD_SCALE)
CARD_X, CARD_Y = (1080 - CARD_W) // 2, 380
RADIUS = 44

# Source moments (seconds), read off the burned-in captions and the audio
HEY_SIRI = 0.75        # "היי Siri"
CALL = 2.6             # "תתקשרי"
NAME = 3.8             # "ס-שה נייד" (loudest word)
SIRI_REPLY = 6.4       # Siri: "שיחת טלפון אל ס-שה נייד"
SIRI_NAME = 7.75
GAP = (8.8, 9.9)       # dead air while the call connects - cut
RING = 10.7            # ringback tone
LAUGH = 11.8           # she turns to the camera laughing
LAST = 12.45
HOLD = 1.3             # freeze on the laugh


class SiriReel:
    W, H, fps = 1080, 1920, 30

    def __init__(self, source):
        self.source = source
        self.src_audio = load_audio(source)
        tm = TimeMap(len(self.src_audio) / 48000, xfade=0.0)
        tm.add_clip(0.0, GAP[0], soft=False)
        tm.add_clip(GAP[1], LAST, soft=False)
        tm.add_freeze(LAST, HOLD)
        self.tm = tm
        self.layers = []
        self.sfx = []
        self.card_alpha = rounded_rect(CARD_W, CARD_H, RADIUS, (255, 255, 255, 255))[..., 3:4].astype(np.float32) / 255

    def o(self, src_t):
        return self.tm.src_to_out(src_t)

    @property
    def duration(self):
        return self.tm.duration

    def open_ctx(self):
        return {"src": SourceReader(self.source)}

    def frame_at(self, t, ctx):
        src = blend_sources(ctx["src"], self.tm.sources_at(t))
        if src.shape[:2] != (SRC_H, SRC_W):
            src = cv2.resize(src, (SRC_W, SRC_H), interpolation=cv2.INTER_AREA)
        # background: the same picture, cover-scaled, blurred and dimmed
        s = self.H / SRC_H
        bg = cv2.resize(src, (int(SRC_W * s), self.H), interpolation=cv2.INTER_AREA)
        x0 = (bg.shape[1] - self.W) // 2
        bg = cv2.GaussianBlur(bg[:, x0:x0 + self.W], (0, 0), 28)
        frame = (bg.astype(np.float32) * 0.42).astype(np.uint8)
        # foreground card with the virtual camera
        M, z = self.camera.matrix(t)
        fg = self.camera.warp(src, M, z)
        fg = cv2.resize(fg, (CARD_W, CARD_H), interpolation=cv2.INTER_AREA).astype(np.float32)
        roi = frame[CARD_Y:CARD_Y + CARD_H, CARD_X:CARD_X + CARD_W].astype(np.float32)
        a = self.card_alpha
        frame[CARD_Y:CARD_Y + CARD_H, CARD_X:CARD_X + CARD_W] = (fg * a + roi * (1 - a) + 0.5).astype(np.uint8)
        info = {"mask": lambda: None}
        for layer in self.layers:
            layer(frame, t, ctx, info)
        return frame

    def prepare_audio(self, workdir):
        os.makedirs(workdir, exist_ok=True)
        raw = os.path.join(workdir, "audio_raw.wav")
        self.audio_path = os.path.join(workdir, "audio.wav")
        track = mix_sfx(build_audio(self.tm, self.src_audio), self.sfx)
        write_wav(raw, track)
        loudnorm_file(raw, self.audio_path)


def build(source):
    p = SiriReel(source)
    o = p.o

    cam = Camera(SRC_W, SRC_H, focus=(120, 640))
    cam.intro = (0.6, 1.15)                                   # hook: quick pull-back on frame 1
    cam.set_base(0, o(GAP[0]), 1.0)
    cam.set_base(o(GAP[1]), o(RING), 1.0, 1.10)               # slow push while the call connects
    cam.set_base(o(RING), p.duration, 1.10)
    cam.punch(o(HEY_SIRI), 0.05, hold=0.4)
    cam.punch(o(CALL), 0.06, hold=0.5)
    cam.punch(o(NAME), 0.09, attack=0.08, hold=0.6)
    cam.shake(o(NAME), 0.25, 8)
    cam.punch(o(SIRI_REPLY), 0.05, hold=0.8)
    cam.punch(o(SIRI_NAME), 0.08, attack=0.08, hold=0.5)
    cam.shake(o(RING), 0.3, 6)
    cam.punch(o(LAUGH), 0.14, attack=0.10, hold=10)          # land on the laugh and stay through the freeze
    p.camera = cam

    t_freeze = o(LAST)
    hook = headline([("ככה", 0.15), ("מדברים", 0.40), ("עם", 0.60), ("סירי:", 0.80)], y_base=245,
                    t_out=t_freeze - 0.05, size=112)
    payoff = headline([("וזה", t_freeze + 0.15), ("עובד.", t_freeze + 0.45)], y_base=250,
                      t_out=1e9, size=130)
    p.layers += [hook, payoff]

    p.sfx += [
        (0.0, sfx.whoosh(0.5, 400, 5000), -18),
        (o(HEY_SIRI) - 0.05, sfx.shimmer(0.7), -26),          # Siri "listening" sparkle
        (o(CALL) - 0.05, sfx.pop(), -26),
        (o(NAME) - 0.02, sfx.thud(0.35), -20),
        (o(SIRI_REPLY) - 0.05, sfx.pop(1200, 700, 0.07), -26),
        (o(GAP[0]) - 0.15, sfx.whoosh(0.35, 600, 3500), -22),  # covers the jump cut
        (o(LAUGH) - 0.08, sfx.whoosh(0.4, 500, 4000), -20),
        (t_freeze, sfx.pop(800, 450, 0.1), -18),
        (t_freeze + 0.1, sfx.shimmer(1.2), -22),
        (t_freeze + 0.45, sfx.pop(1000, 500, 0.09), -20),
    ]
    p.prepare_audio(WORK)
    return p


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "input", "siri.mov")
    p = build(src)
    print(p.tm.describe())
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, "siri_v1.mp4")
    render(p, out)
    print("rendered", out)
    for page in contact_sheet(out, os.path.join(WORK, "contact_v1"), per_sec=4):
        print("contact", page)
