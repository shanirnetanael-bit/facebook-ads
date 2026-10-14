"""Motion-graphics layers. Each layer is a callable layer(frame, t, ctx, info) -> frame, with an
`active(t)` window. Wrap a layer in Behind(...) to place it behind the person (uses the person mask)."""
import math

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .text import over, render_text, rounded_rect


def clamp01(x):
    return min(max(x, 0.0), 1.0)


def ramp(t, t0, d):
    return clamp01((t - t0) / d) if d > 0 else float(t >= t0)


def ease_out(x):
    x = clamp01(x)
    return 1 - (1 - x) ** 3


def ease_in(x):
    x = clamp01(x)
    return x * x * x


def ease_out_back(x, s=1.70158):
    x = clamp01(x)
    return 1 + (s + 1) * (x - 1) ** 3 + s * (x - 1) ** 2


def rotate_rgba(rgba, deg):
    h, w = rgba.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    c, s = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * s + w * c) + 2, int(h * c + w * s) + 2
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(rgba, M, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))


class Layer:
    t_in = 0.0
    t_out = 1e9
    tail = 0.4

    def active(self, t):
        return self.t_in - 0.05 <= t < self.t_out + self.tail

    def __call__(self, frame, t, ctx, info):
        if self.active(t):
            self.draw(frame, t, info)
        return frame


class Behind(Layer):
    """Draw `inner` between the background and the person."""

    def __init__(self, inner):
        self.inner = inner

    def active(self, t):
        return self.inner.active(t)

    def __call__(self, frame, t, ctx, info):
        if not self.active(t):
            return frame
        m = info["mask"]()
        if m is None:
            return self.inner(frame, t, ctx, info)
        orig = frame.copy()
        self.inner.draw(frame, t, info)
        m3 = m[..., None]
        return (frame.astype(np.float32) * (1 - m3) + orig.astype(np.float32) * m3 + 0.5).astype(np.uint8)


class Title(Layer):
    """Big kinetic headline. parts = [(text, rgba, t_appear)], laid out right-to-left on one line.
    Each part rises + fades in on its own word; everything lifts away at t_out."""

    def __init__(self, parts, font, size, y_base, t_out, cx=540, weight=900, space=None, rise=70,
                 stroke=0, stroke_color=(0, 0, 0, 255), shadow=(0, 0, 0, 0)):
        self.parts = []
        for text, col, t0 in parts:
            patch, meta = render_text(text, font, size, col, weight, shadow=shadow, shadow_blur=10, shadow_dy=6,
                                      stroke=stroke, stroke_color=stroke_color)
            self.parts.append((patch, meta, t0))
        space = size * 0.28 if space is None else space
        total = sum(m["ink_w"] for _, m, _ in self.parts) + space * (len(self.parts) - 1)
        x = cx + total / 2
        self.pos = []
        for patch, meta, _ in self.parts:
            self.pos.append((x - meta["ink_w"], y_base))           # ink left, baseline
            x -= meta["ink_w"] + space
        self.t_in, self.t_out, self.rise = min(p[2] for p in self.parts), t_out, rise
        self.tail = 0.3

    def draw(self, frame, t, info):
        out_p = ease_in(ramp(t, self.t_out, 0.28))
        for (patch, meta, t0), (x0, yb) in zip(self.parts, self.pos):
            if t < t0 - 0.02:
                continue
            p = ease_out(ramp(t, t0 - 0.02, 0.32))
            a = ramp(t, t0 - 0.02, 0.16) * (1 - out_p)
            dy = (1 - p) * self.rise - out_p * 40
            sc = 1.08 - 0.08 * p
            ax = (meta["ink_x"] + meta["ink_w"] / 2) / patch.shape[1]
            ay = meta["baseline"] / patch.shape[0]
            over(frame, patch, x0 + meta["ink_w"] / 2, yb + dy, opacity=a, scale=sc, anchor=(ax, ay))


class NameCard(Layer):
    """Lower-third style name card: gold bar grows, navy card wipes open right->left, text slides in."""

    def __init__(self, name, sub, font, t_in, t_out, right=440, top=300, navy=(11, 27, 52, 235),
                 gold=(212, 175, 55, 255), white=(255, 255, 255, 255)):
        self.name, self.nm = render_text(name, font, 64, white, 800)
        self.sub, self.sm = render_text(sub, font, 36, gold, 600)
        self.t_in, self.t_out = t_in, t_out
        self.right, self.top = right, top
        self.w = int(max(self.nm["ink_w"], self.sm["ink_w"]) + 72)
        self.h = 150
        self.navy, self.gold = navy, gold
        self.tail = 0.35

    def draw(self, frame, t, info):
        close = ease_in(ramp(t, self.t_out, 0.3))
        bar_p = ease_out(ramp(t, self.t_in, 0.2)) * (1 - close)
        wipe = ease_out(ramp(t, self.t_in + 0.1, 0.35)) * (1 - close)
        if bar_p <= 0:
            return
        bh = max(2, int(self.h * bar_p))
        over(frame, rounded_rect(8, bh, 3, self.gold), self.right - 4, self.top + self.h / 2, anchor=(0.5, 0.5))
        cw = int(self.w * wipe)
        if cw > 24:
            over(frame, rounded_rect(cw, self.h, 18, self.navy), self.right - 12, self.top, anchor=(1.0, 0.0))
        ta = ramp(t, self.t_in + 0.25, 0.2) * (1 - close)
        dx = (1 - ease_out(ramp(t, self.t_in + 0.25, 0.35))) * 40
        xr = self.right - 40 + dx
        for patch, meta, base in ((self.name, self.nm, self.top + 72), (self.sub, self.sm, self.top + 122)):
            over(frame, patch, xr, base, opacity=ta,
                 anchor=((meta["ink_x"] + meta["ink_w"]) / patch.shape[1], meta["baseline"] / patch.shape[0]))


class Stamp(Layer):
    """Rubber stamp that slams onto the frame (pair it with Camera.shake + a thud)."""

    def __init__(self, text, font, t_in, t_out, x, y, color=(11, 27, 52, 255), angle=-12, size=120):
        txt, m = render_text(text, font, size, color, 900)
        w, h = int(m["ink_w"] + 90), int(size * 1.35)
        img = Image.new("RGBA", (w * 4, h * 4), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle((6 * 4, 6 * 4, (w - 6) * 4, (h - 6) * 4), radius=22 * 4, outline=color, width=9 * 4)
        box = np.asarray(img.resize((w, h), Image.LANCZOS)).copy()
        over_rgba = box.astype(np.float32)
        # place text centred into the box (RGBA over RGBA)
        th, tw = txt.shape[:2]
        ox = int(w / 2 - (m["ink_x"] + m["ink_w"] / 2))
        oy = int(h / 2 - (m["baseline"] - size * 0.36))
        canvas = np.zeros((h, w, 4), np.float32)
        ys, xs = slice(max(oy, 0), min(oy + th, h)), slice(max(ox, 0), min(ox + tw, w))
        canvas[ys, xs] = txt[ys.start - oy:ys.stop - oy, xs.start - ox:xs.stop - ox]
        a1, a2 = over_rgba[..., 3:] / 255, canvas[..., 3:] / 255
        a = a2 + a1 * (1 - a2)
        rgb = (canvas[..., :3] * a2 + over_rgba[..., :3] * a1 * (1 - a2)) / np.maximum(a, 1e-6)
        self.patch = rotate_rgba(np.dstack([rgb, a * 255]).clip(0, 255).astype(np.uint8), angle)
        self.t_in, self.t_out, self.x, self.y = t_in, t_out, x, y
        self.tail = 0.3

    def draw(self, frame, t, info):
        slam = ramp(t, self.t_in, 0.13)
        sc = 2.0 - 1.0 * ease_in(slam)
        if slam >= 1:                                   # tiny settle after impact
            u = t - self.t_in - 0.13
            sc = 1.0 - 0.04 * math.exp(-u * 18) * math.sin(u * 40)
        a = ramp(t, self.t_in, 0.06) * 0.93 * (1 - ramp(t, self.t_out, 0.25))
        over(frame, self.patch, self.x, self.y, opacity=a, scale=sc)


class ShineCard(Layer):
    """Navy card with gold words appearing one by one, then a light sweep across the text."""

    def __init__(self, parts, font, size, cy, t_out, t_shine, cx=540, navy=(11, 27, 52, 240), gold=(212, 175, 55, 255)):
        self.title = Title([(p, gold, t0) for p, t0 in parts], font, size, cy + size * 0.36, t_out, cx=cx, rise=36)
        total = self.title.pos[0][0] + self.title.parts[0][1]["ink_w"] - self.title.pos[-1][0]
        self.card_w, self.card_h = int(total + 110), int(size * 1.55)
        self.card = rounded_rect(self.card_w, self.card_h, 34, navy)
        self.cx, self.cy = cx, cy
        self.t_in, self.t_out, self.t_shine = self.title.t_in - 0.04, t_out, t_shine
        self.tail = 0.3

    def draw(self, frame, t, info):
        p = ease_out_back(ramp(t, self.t_in, 0.32))
        a = ramp(t, self.t_in, 0.15) * (1 - ramp(t, self.t_out, 0.25))
        over(frame, self.card, self.cx, self.cy, opacity=a, scale=0.85 + 0.15 * p)
        self.title.draw(frame, t, info)
        # light sweep: a soft diagonal band moving right->left over the card
        u = ramp(t, self.t_shine, 0.55)
        if 0 < u < 1:
            x0 = int(self.cx - self.card_w / 2)
            y0 = int(self.cy - self.card_h / 2)
            H, W = frame.shape[:2]
            xs = np.arange(self.card_w)[None, :]
            ys = np.arange(self.card_h)[:, None]
            band_c = (1 - u) * (self.card_w + 200) - 100
            d = (xs + 0.35 * ys) - band_c
            band = np.exp(-(d / 38.0) ** 2) * 0.55 * a
            card_a = self.card[..., 3].astype(np.float32) / 255
            band = band * card_a
            ya, yb, xa, xb = max(y0, 0), min(y0 + self.card_h, H), max(x0, 0), min(x0 + self.card_w, W)
            roi = frame[ya:yb, xa:xb].astype(np.float32)
            b = band[ya - y0:yb - y0, xa - x0:xb - x0, None]
            frame[ya:yb, xa:xb] = np.clip(roi + (255 - roi) * b, 0, 255).astype(np.uint8)


def check_badge(d, gold=(212, 175, 55, 255), navy=(11, 27, 52, 255)):
    s = 4
    img = Image.new("RGBA", (d * s, d * s), (0, 0, 0, 0))
    g = ImageDraw.Draw(img)
    g.ellipse((0, 0, d * s - 1, d * s - 1), fill=gold)
    pts = [(0.27, 0.52), (0.43, 0.68), (0.74, 0.35)]
    g.line([(x * d * s, y * d * s) for x, y in pts], fill=navy, width=int(d * s * 0.12), joint="curve")
    return np.asarray(img.resize((d, d), Image.LANCZOS)).copy()


class Checklist(Layer):
    """Header + 2x2 grid of chips (RTL order: top-right, top-left, bottom-right, bottom-left);
    each chip pops in on its word and its gold check lands a beat later."""

    def __init__(self, header, items, font, t_header, t_out, top=1170, navy=(11, 27, 52, 235),
                 gold=(212, 175, 55, 255), white=(255, 255, 255, 255), chip_w=400, item_size=54):
        self.hdr, self.hm = render_text(header, font, 54, gold, 800, shadow=(0, 0, 0, 120), shadow_blur=8)
        self.items = []
        for text, t0 in items:
            self.items.append((render_text(text, font, item_size, white, 700), t0))
        self.badge = check_badge(58, gold, navy[:3] + (255,))
        self.chip_w, self.chip_h = chip_w, 100
        self.chip = rounded_rect(self.chip_w, self.chip_h, 50, navy)
        self.hdr_pill = rounded_rect(int(self.hm["ink_w"] + 70), 84, 42, navy)
        self.t_in, self.t_out, self.t_header = t_header, t_out, t_header
        self.top = top
        self.tail = 0.35

    def draw(self, frame, t, info):
        fade = 1 - ramp(t, self.t_out, 0.3)
        ha = ramp(t, self.t_header, 0.2) * fade
        hp = ease_out_back(ramp(t, self.t_header, 0.3))
        over(frame, self.hdr_pill, 540, self.top, opacity=ha, scale=0.85 + 0.15 * hp)
        over(frame, self.hdr, 540, self.top + 18, opacity=ha, scale=0.85 + 0.15 * hp,
             anchor=((self.hm["ink_x"] + self.hm["ink_w"] / 2) / self.hdr.shape[1], self.hm["baseline"] / self.hdr.shape[0]))
        cols = [540 + self.chip_w / 2 + 12, 540 - self.chip_w / 2 - 12]
        for k, ((patch_meta), t0) in enumerate(self.items):
            patch, meta = patch_meta
            if t < t0 - 0.06:
                continue
            cx, cy = cols[k % 2], self.top + 125 + (k // 2) * 125
            p = ease_out_back(ramp(t, t0 - 0.06, 0.3))
            a = ramp(t, t0 - 0.06, 0.14) * fade
            sc = 0.75 + 0.25 * p
            over(frame, self.chip, cx, cy, opacity=a, scale=sc)
            # text right-aligned next to the badge (badge sits at the chip's right end)
            bx = cx + (self.chip_w / 2 - 50) * sc
            tx = bx - 46 * sc
            over(frame, patch, cx + (tx - cx), cy + 18 * sc, opacity=a, scale=sc,
                 anchor=((meta["ink_x"] + meta["ink_w"]) / patch.shape[1], meta["baseline"] / patch.shape[0]))
            bp = ease_out_back(ramp(t, t0 + 0.08, 0.25), 2.5)
            if bp > 0:
                over(frame, self.badge, bx, cy, opacity=fade, scale=sc * bp)


class EndCard(Layer):
    """Freeze-frame outro: picture blurs and darkens to navy; name, gold rule and subtitle rise in,
    then an optional call to action: a white line + a gold button that breathes gently."""

    def __init__(self, name, sub, font, t_in, cta=None, navy=(11, 27, 52), gold=(212, 175, 55, 255),
                 white=(255, 255, 255, 255)):
        self.name, self.nm = render_text(name, font, 100, white, 800)
        self.sub, self.sm = render_text(sub, font, 50, gold, 600)
        self.t_in = t_in
        self.navy = np.array(navy[::-1], np.float32)
        self.gold = gold
        self.cta = None
        if cta:
            line1, line2 = cta
            self.cta = (render_text(line1, font, 62, white, 700),
                        render_text(line2, font, 62, navy + (255,), 800))
            self.btn = rounded_rect(int(self.cta[1][1]["ink_w"] + 110), 118, 59, gold)
        self.base_y = 720 if cta else 900
        self.tail = 0

    def active(self, t):
        return t >= self.t_in

    @staticmethod
    def _centre(meta, patch):
        return ((meta["ink_x"] + meta["ink_w"] / 2) / patch.shape[1], meta["baseline"] / patch.shape[0])

    def draw(self, frame, t, info):
        u = ease_out(ramp(t, self.t_in, 0.45))
        blurred = cv2.GaussianBlur(frame, (0, 0), 1 + 22 * u)
        f = blurred.astype(np.float32) * (1 - 0.82 * u) + self.navy * 0.82 * u
        frame[:] = np.clip(f, 0, 255).astype(np.uint8)
        y = self.base_y
        for patch, meta, base, d in ((self.name, self.nm, y, 0.25), (self.sub, self.sm, y + 135, 0.55)):
            p = ease_out(ramp(t, self.t_in + d, 0.45))
            over(frame, patch, 540, base + (1 - p) * 40, opacity=ramp(t, self.t_in + d, 0.25),
                 anchor=self._centre(meta, patch))
        rp = ease_out(ramp(t, self.t_in + 0.4, 0.5))
        if rp > 0.02:
            over(frame, rounded_rect(max(4, int(300 * rp)), 6, 3, self.gold), 540, y + 50)
        if self.cta:
            (l1, m1), (l2, m2) = self.cta
            t1 = self.t_in + 0.85
            p = ease_out(ramp(t, t1, 0.45))
            over(frame, l1, 540, y + 330 + (1 - p) * 40, opacity=ramp(t, t1, 0.25), anchor=self._centre(m1, l1))
            t2 = t1 + 0.3
            bp = ease_out_back(ramp(t, t2, 0.4))
            if bp > 0:
                breathe = 1 + 0.025 * math.sin(2 * math.pi * 0.9 * max(t - t2 - 0.4, 0))
                sc = (0.8 + 0.2 * bp) * breathe
                a = ramp(t, t2, 0.2)
                cy = y + 465
                over(frame, self.btn, 540, cy, opacity=a, scale=sc)
                over(frame, l2, 540, cy + 0.36 * 62 * sc, opacity=a, scale=sc, anchor=self._centre(m2, l2))


def _rgba_canvas(w, h, draw_fn, s=4):
    """Supersampled Pillow drawing -> RGBA uint8 patch."""
    img = Image.new("RGBA", (w * s, h * s), (0, 0, 0, 0))
    draw_fn(ImageDraw.Draw(img), s)
    return np.asarray(img.resize((w, h), Image.LANCZOS)).copy()


class Clock(Layer):
    """Analogue clock that pops in, its hands sweep fast (time running out), then it pops out.
    Pair with ticks/whoosh in the sound design."""

    def __init__(self, t_in, t_out, x, y, d=230, navy=(11, 27, 52, 240), gold=(212, 175, 55, 255),
                 white=(255, 255, 255, 255), turns=1.5):
        self.face = _rgba_canvas(d, d, lambda g, s: (
            g.ellipse((0, 0, d * s - 1, d * s - 1), fill=gold),
            g.ellipse((10 * s, 10 * s, (d - 10) * s, (d - 10) * s), fill=navy),
            [g.line([(d * s / 2 + math.sin(a) * (d / 2 - 24) * s, d * s / 2 - math.cos(a) * (d / 2 - 24) * s),
                     (d * s / 2 + math.sin(a) * (d / 2 - (40 if k % 3 == 0 else 32)) * s,
                      d * s / 2 - math.cos(a) * (d / 2 - (40 if k % 3 == 0 else 32)) * s)],
                    fill=white, width=int((6 if k % 3 == 0 else 3) * s))
             for k, a in enumerate(np.linspace(0, 2 * math.pi, 12, endpoint=False))]))
        self.d, self.x, self.y = d, x, y
        self.gold, self.white = gold, white
        self.t_in, self.t_out, self.turns = t_in, t_out, turns
        self.tail = 0.3

    def _hand(self, length, width, angle, color):
        d = self.d
        c = d / 2
        return _rgba_canvas(d, d, lambda g, s: g.line(
            [(c * s, c * s), ((c + math.sin(angle) * length) * s, (c - math.cos(angle) * length) * s)],
            fill=color, width=int(width * s), joint="curve"))

    def draw(self, frame, t, info):
        p = ease_out_back(ramp(t, self.t_in, 0.35), 2.2)
        out = ease_in(ramp(t, self.t_out, 0.25))
        sc = (0.4 + 0.6 * p) * (1 - 0.6 * out)
        a = ramp(t, self.t_in, 0.12) * (1 - out)
        if a <= 0:
            return
        u = ease_out(ramp(t, self.t_in, max(self.t_out - self.t_in, 0.3)))
        wob = math.sin((t - self.t_in) * 9) * 4 * (1 - ramp(t, self.t_in, 0.8))
        over(frame, rotate_rgba(self.face, wob) if abs(wob) > 0.2 else self.face, self.x, self.y, opacity=a, scale=sc)
        r = self.d / 2
        over(frame, self._hand(r * 0.45, 9, 2 * math.pi * (0.33 + u * self.turns / 12), self.white), self.x, self.y,
             opacity=a, scale=sc)
        over(frame, self._hand(r * 0.68, 6, 2 * math.pi * (0.0 + u * self.turns), self.gold), self.x, self.y,
             opacity=a, scale=sc)
        over(frame, rounded_rect(18, 18, 9, self.gold), self.x, self.y, opacity=a, scale=sc)


class Timeline(Layer):
    """RTL timeline: a gold line draws from right to left; each stop (label, t) pops a dot and a pill.
    The last stop can be a '?' that wobbles (more changes to come)."""

    def __init__(self, stops, font, t_out, y=300, x_right=930, x_left=150, navy=(11, 27, 52, 235),
                 gold=(212, 175, 55, 255), white=(255, 255, 255, 255)):
        self.stops = []
        n = len(stops)
        for k, (label, t0) in enumerate(stops):
            x = x_right - (x_right - x_left) * (k + 0.5) / n
            patch = render_text(label, font, 64, white if label != "?" else navy, 800)
            pill = rounded_rect(int(patch[1]["ink_w"] + 60), 96, 48, navy if label != "?" else gold)
            self.stops.append((x, t0, patch, pill))
        self.y, self.x_right, self.x_left = y, x_right, x_left
        self.gold = gold
        self.t_in, self.t_out = stops[0][1] - 0.25, t_out
        self.dot = _rgba_canvas(34, 34, lambda g, s: g.ellipse((0, 0, 34 * s - 1, 34 * s - 1), fill=gold))
        self.tail = 0.35

    def draw(self, frame, t, info):
        fade = 1 - ramp(t, self.t_out, 0.3)
        # the line reaches each stop as it is said
        reach = self.x_right
        prev_x, prev_t = self.x_right + 40, self.t_in
        for x, t0, _, _ in self.stops:
            if t >= t0:
                prev_x, prev_t = x, t0
                reach = x
            else:
                reach = prev_x + (x - prev_x) * ease_out(ramp(t, t0 - 0.3, 0.3))
                break
        else:
            reach = self.stops[-1][0] - 40 * ease_out(ramp(t, self.stops[-1][1], 0.3))
        w = int(self.x_right + 40 - reach)
        if w > 4:
            over(frame, rounded_rect(w, 8, 4, self.gold), self.x_right + 40, self.y + 80, opacity=fade, anchor=(1.0, 0.5))
        for x, t0, (patch, meta), pill in self.stops:
            if t < t0 - 0.05:
                continue
            p = ease_out_back(ramp(t, t0 - 0.05, 0.32), 2.4)
            a = ramp(t, t0 - 0.05, 0.12) * fade
            over(frame, self.dot, x, self.y + 80, opacity=a, scale=p)
            wob = 0
            if meta["ink_w"] < 60:                           # the '?' keeps wobbling
                wob = math.sin((t - t0) * 7) * 0.06
            sc = (0.6 + 0.4 * p) * (1 + wob)
            over(frame, pill, x, self.y, opacity=a, scale=sc)
            over(frame, patch, x, self.y + 22 * sc, opacity=a, scale=sc,
                 anchor=((meta["ink_x"] + meta["ink_w"] / 2) / patch.shape[1], meta["baseline"] / patch.shape[0]))


class NumberFlip(Layer):
    """'<n> <label>' card; at t_flip the number is crossed out, drops away and the new number lands
    with an overshoot (e.g. 3 -> 4 activities)."""

    def __init__(self, n_from, n_to, label, font, t_in, t_flip, t_out, cx=540, cy=300, navy=(11, 27, 52, 240),
                 gold=(212, 175, 55, 255), white=(255, 255, 255, 255)):
        self.lab = render_text(label, font, 96, white, 800)
        self.n0 = render_text(str(n_from), font, 150, gold, 900)
        self.n1 = render_text(str(n_to), font, 150, gold, 900)
        lw = self.lab[1]["ink_w"]
        self.num_w = max(self.n0[1]["ink_w"], self.n1[1]["ink_w"])
        total = lw + 40 + self.num_w
        self.card = rounded_rect(int(total + 120), 210, 42, navy)
        self.num_x = cx + total / 2 - self.num_w / 2          # number on the right (RTL)
        self.lab_x = cx - total / 2 + lw / 2
        self.cx, self.cy = cx, cy
        self.gold = gold
        self.t_in, self.t_flip, self.t_out = t_in, t_flip, t_out
        self.tail = 0.3

    @staticmethod
    def _c(pm):
        patch, m = pm
        return ((m["ink_x"] + m["ink_w"] / 2) / patch.shape[1], (m["baseline"]) / patch.shape[0])

    def draw(self, frame, t, info):
        out = ramp(t, self.t_out, 0.25)
        p = ease_out_back(ramp(t, self.t_in, 0.32))
        a = ramp(t, self.t_in, 0.12) * (1 - out)
        if a <= 0:
            return
        sc = 0.8 + 0.2 * p
        over(frame, self.card, self.cx, self.cy, opacity=a, scale=sc)
        base = self.cy + 50
        over(frame, self.lab[0], self.lab_x, base, opacity=a, anchor=self._c(self.lab))
        # cross-out sweep just before the flip, then old number falls, new one drops in
        strike = ease_out(ramp(t, self.t_flip - 0.25, 0.2))
        drop = ease_in(ramp(t, self.t_flip, 0.22))
        if drop < 1:
            over(frame, self.n0[0], self.num_x, base + drop * 120, opacity=a * (1 - drop), anchor=self._c(self.n0))
            if strike > 0:
                sw = int((self.num_w + 30) * strike)
                if sw > 4:
                    over(frame, rounded_rect(sw, 12, 6, (255, 255, 255, 255)), self.num_x + (self.num_w + 30) / 2,
                         self.cy - 4, opacity=a * (1 - drop), anchor=(1.0, 0.5))
        land = ramp(t, self.t_flip + 0.08, 0.3)
        if land > 0:
            s2 = 1.6 - 0.6 * ease_out_back(land, 2.0)
            over(frame, self.n1[0], self.num_x, base - (1 - ease_out(land)) * 90, opacity=a * ramp(t, self.t_flip + 0.08, 0.1),
                 scale=s2, anchor=self._c(self.n1))


class DownArrows(Layer):
    """Three gold chevrons cascading downward ('details below'), looping until t_out."""

    def __init__(self, t_in, t_out, x=540, y=1530, gold=(212, 175, 55, 255), navy=(11, 27, 52, 255), size=96):
        s_ = size
        pts = [(12, 10), (s_ / 2, s_ * 0.6 - 12), (s_ - 12, 10)]
        self.chev = _rgba_canvas(s_, int(s_ * 0.6), lambda g, s: (
            g.line([(x * s, y * s) for x, y in pts], fill=navy, width=int(22 * s), joint="curve"),
            g.line([(x * s, y * s) for x, y in pts], fill=gold, width=int(12 * s), joint="curve")))
        self.t_in, self.t_out, self.x, self.y, self.size = t_in, t_out, x, y, size
        self.tail = 0.3

    def draw(self, frame, t, info):
        fade = ramp(t, self.t_in, 0.2) * (1 - ramp(t, self.t_out, 0.25))
        for k in range(3):
            ph = ((t - self.t_in) * 1.6 - k * 0.22) % 1.0
            a = math.sin(math.pi * ph) * fade
            over(frame, self.chev, self.x, self.y + k * self.size * 0.55 + ph * 18, opacity=a)
