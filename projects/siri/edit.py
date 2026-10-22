"""Siri reel (comedy) - driving, she says the name deliberately mangled ("ס-שה") so Siri will dial the right
person; Siri repeats it, the call rings, she laughs to camera.
The source already has burned-in captions, so this edit adds framing, headlines, pacing and sound.

Format: full-screen Instagram Reel, 1080x1920 (9:16), 30 fps. The 1080x1500 source is scaled to fill the
height and a moving crop window pans across it: on her face while she talks, over to the phone while it
dials, and back to her when she laughs to camera.

Retention design (open loop -> anticipation -> payoff):
  0.0s   "ככה מדברים עם סירי:"  - promise of a method, visible on frame 1 (also the cover)
  ~3.8s  the "method": the mangled name, punched in
  ~5.0s  "זה יעבוד?"            - asked in the silence before Siri answers; the answer is held back
  ~6.4s  Siri repeats the mangled name, 📞 pops in and rings with the ringback
  ring   the punchline: "אלא אם את רוצה / שהיא תחייג לנטשה." (נטשה rhymes with סשה)
  end    she laughs to camera, freeze on the smile (peak-end)

Run from the repo root:  python -m projects.siri.edit <source.mov>
"""
import math
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from engine import sfx
from engine.audio import build_audio, load_audio, loudnorm_file, mix_sfx, write_wav
from engine.camera import Camera, smooth
from engine.fx import Layer, ease_out_back, ramp, rotate_rgba
from engine.render import SourceReader, blend_sources, contact_sheet, render
from engine.style import headline
from engine.text import over
from engine.timemap import TimeMap

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORK = os.path.join(ROOT, "work", "siri")
OUT = os.path.join(ROOT, "out")
EMOJI_FONT = "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"

SRC_W, SRC_H = 1080, 1500
SCALE = 1920 / SRC_H                     # fill the reel height
WIDE = int(round(SRC_W * SCALE))         # 1382 px wide before the crop window
MAX_PAN = WIDE - 1080

# Source moments (seconds), read off the burned-in captions and the audio
HEY_SIRI = 0.75        # "היי Siri"
CALL = 2.6             # "תתקשרי"
NAME = 3.8             # "ס-שה נייד" (loudest word)
SIRI_REPLY = 6.4       # Siri: "שיחת טלפון אל ס-שה נייד"
SIRI_NAME = 7.75
GAP = (8.8, 9.9)       # dead air while the call connects - cut
RING = 10.7            # ringback tone (425 Hz) - in the original audio
LAUGH = 11.8           # she turns to the camera laughing
LAST = 12.45
HOLD = 2.8             # freeze on the laugh (time to read the tip)
ASK = 4.95             # right after the mangled "ס-שה נייד", before Siri answers

FACE = (380, 650)      # zoom centre while she talks (source px)
FACE_TURNED = (120, 680)


def keyframes(t, keys):
    """Smooth interpolation through [(t, value)]."""
    if t <= keys[0][0]:
        return keys[0][1]
    for (t0, v0), (t1, v1) in zip(keys, keys[1:]):
        if t < t1:
            return v0 + (v1 - v0) * smooth((t - t0) / (t1 - t0))
    return keys[-1][1]


def emoji_patch(ch, size_px):
    f = ImageFont.truetype(EMOJI_FONT, 109)            # bitmap colour font: only 109 px exists
    img = Image.new("RGBA", (160, 160), (0, 0, 0, 0))
    ImageDraw.Draw(img).text((80, 80), ch, font=f, embedded_color=True, anchor="mm")
    img = img.crop(img.getbbox()).resize((size_px, size_px), Image.LANCZOS)
    pad = size_px // 3
    canvas = Image.new("RGBA", (size_px + 2 * pad, size_px + 2 * pad), (0, 0, 0, 0))
    canvas.paste(img, (pad, pad), img)
    rgba = np.asarray(canvas).copy()
    # soft dark halo, like the headlines
    a = rgba[..., 3].astype(np.float32) / 255
    halo = cv2.GaussianBlur(a, (0, 0), size_px * 0.06) * 0.55
    out = np.zeros_like(rgba, dtype=np.float32)
    out[..., 3] = halo
    ca = a
    a_o = ca + halo * (1 - ca)
    out[..., :3] = rgba[..., :3] * (ca / np.maximum(a_o, 1e-6))[..., None]
    out[..., 3] = a_o * 255
    return np.clip(out, 0, 255).astype(np.uint8)


class RingingEmoji(Layer):
    """📞 pops in, then rings (fast wiggle bursts) until t_out."""

    def __init__(self, x, y, t_in, t_out, ring_from, size=210):
        self.patch = emoji_patch("📞", size)
        self.x, self.y, self.t_in, self.t_out, self.ring_from = x, y, t_in, t_out, ring_from
        self.tail = 0.25

    def draw(self, frame, t, info):
        sc = ease_out_back(ramp(t, self.t_in, 0.35)) * (1 - ramp(t, self.t_out, 0.2))
        ang = 0.0
        if t >= self.ring_from:
            ph = (t - self.ring_from) % 1.0                      # one burst per second
            if ph < 0.45:
                ang = 14 * math.sin(2 * math.pi * 14 * ph) * (1 - ph / 0.45)
        else:
            ang = 4 * math.sin(2 * math.pi * 1.2 * (t - self.t_in))   # idle sway
        dy = 6 * math.sin(2 * math.pi * 0.9 * (t - self.t_in))
        over(frame, rotate_rgba(self.patch, ang), self.x, self.y + dy, opacity=min(1.0, sc * 1.5), scale=max(sc, 0.01))


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
        g = np.linspace(1, 0, 760, dtype=np.float32) ** 1.6 * 0.45      # top shade so the yellow reads
        self.shade = 1 - g[:, None, None]

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
        self.camera.focus = (keyframes(t, self.focus_keys), keyframes(t, self.focus_y_keys))
        M, z = self.camera.matrix(t)
        src = self.camera.warp(src, M, z)
        wide = cv2.resize(src, (WIDE, self.H), interpolation=cv2.INTER_CUBIC)
        x0 = int(round(keyframes(t, self.pan_keys)))
        frame = wide[:, x0:x0 + self.W]
        blur = cv2.GaussianBlur(frame, (0, 0), 1.2)
        frame = cv2.addWeighted(frame, 1.35, blur, -0.35, 0)            # light sharpen after the upscale
        top = frame[:len(self.shade)].astype(np.float32) * self.shade
        frame[:len(self.shade)] = top.astype(np.uint8)
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
    t_freeze = o(LAST)

    # crop window: on her face -> over to the phone while it dials -> back to her for the laugh
    p.pan_keys = [(0, 110), (o(SIRI_REPLY), 110), (o(SIRI_REPLY) + 0.8, MAX_PAN),
                  (o(LAUGH) - 0.1, MAX_PAN), (o(LAUGH) + 0.45, 0)]
    p.focus_keys = [(0, FACE[0]), (o(LAUGH) - 0.1, FACE[0]), (o(LAUGH) + 0.45, FACE_TURNED[0])]
    p.focus_y_keys = [(0, FACE[1]), (o(LAUGH) - 0.1, FACE[1]), (o(LAUGH) + 0.45, FACE_TURNED[1])]

    cam = Camera(SRC_W, SRC_H, focus=FACE)
    cam.intro = (0.5, 1.12)                                    # quick settle on frame 1 (pattern interrupt)
    cam.set_base(0, o(GAP[0]), 1.0)
    cam.set_base(o(GAP[1]), o(RING), 1.0, 1.08)                # slow push while the call connects
    cam.set_base(o(RING), p.duration, 1.08)
    cam.punch(o(CALL), 0.05, hold=0.5)
    cam.punch(o(NAME), 0.10, attack=0.08, hold=0.7)            # the mangled name
    cam.punch(o(SIRI_NAME), 0.07, attack=0.08, hold=0.5)       # Siri repeats it
    cam.punch(o(LAUGH), 0.12, attack=0.12, hold=10)            # land on the laugh, stay through the freeze
    p.camera = cam

    Y = 470
    tip = o(RING)                                              # the call rings -> the tip
    p.layers += [
        headline([("ככה", 0.0), ("מדברים", 0.25), ("עם", 0.45), ("סירי:", 0.65)], y_base=Y,
                 t_out=o(ASK) - 0.1, size=100),
        headline([("זה", o(ASK)), ("יעבוד?", o(ASK) + 0.25)], y_base=Y, t_out=o(SIRI_REPLY) - 0.05, size=140),
        RingingEmoji(860, 760, t_in=o(SIRI_REPLY), t_out=o(LAUGH) - 0.15, ring_from=o(RING)),
        headline([("אלא", tip), ("אם", tip + 0.15), ("את", tip + 0.3), ("רוצה", tip + 0.45)],
                 y_base=Y - 10, t_out=1e9, size=110),
        headline([("שהיא", tip + 1.1), ("תחייג", tip + 1.3), ("לנטשה.", tip + 1.65)],
                 y_base=Y + 125, t_out=1e9, size=110),
    ]

    # sound: no zoom whooshes - only soft accents on what appears
    p.sfx += [
        (o(HEY_SIRI) - 0.05, sfx.shimmer(0.7), -30),
        (o(SIRI_REPLY), sfx.pop(1100, 600, 0.08), -24),          # 📞 lands
        (o(ASK), sfx.tick(), -24),                               # "זה יעבוד?"
        (tip + 1.65, sfx.pop(1000, 500, 0.09), -20),             # "...לנטשה."
        (tip + 1.7, sfx.shimmer(1.1), -28),
    ]
    p.prepare_audio(WORK)
    return p


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "input", "siri.mov")
    version = os.environ.get("VERSION", "3")
    p = build(src)
    print(p.tm.describe())
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, f"siri_v{version}.mp4")
    render(p, out, crf=15)
    print("rendered", out)
    for page in contact_sheet(out, os.path.join(WORK, f"contact_v{version}"), per_sec=4):
        print("contact", page)
