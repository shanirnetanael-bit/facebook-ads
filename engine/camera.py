"""Virtual camera: zoom / punch-in / slow push / shake, applied as one affine warp to the picture
AND to the person mask, so layers placed "behind" the person stay locked to them."""
import math

import cv2
import numpy as np


def smooth(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def ease_out(x):
    x = min(max(x, 0.0), 1.0)
    return 1 - (1 - x) ** 3


class Camera:
    def __init__(self, W, H, focus):
        self.W, self.H = W, H
        self.focus = focus                  # zoom centre (x, y), e.g. the face
        self.base = []                      # (t0, t1, z0, z1): zoom ramps linearly inside [t0, t1]
        self.base_blend = 0.10              # seconds to glide between consecutive base values
        self.punches = []                   # (t, amount, attack, hold, release)
        self.shakes = []                    # (t, dur, amp_px)
        self.intro = None                   # (dur, from_zoom)

    def set_base(self, t0, t1, z0, z1=None):
        self.base.append((t0, t1, z0, z0 if z1 is None else z1))
        self.base.sort()
        return self

    def punch(self, t, amount, attack=0.12, hold=0.5, release=0.45):
        self.punches.append((t, amount, attack, hold, release))
        return self

    def shake(self, t, dur=0.3, amp=14):
        self.shakes.append((t, dur, amp))
        return self

    def _base(self, t):
        def val(seg, t):
            t0, t1, z0, z1 = seg
            return z0 + (z1 - z0) * min(max((t - t0) / max(t1 - t0, 1e-6), 0), 1)
        if not self.base:
            return 1.0
        z = val(self.base[0], t)
        for prev, seg in zip(self.base, self.base[1:]):
            if t >= seg[0] - self.base_blend / 2:
                k = smooth((t - (seg[0] - self.base_blend / 2)) / self.base_blend)
                z = val(prev, t) * (1 - k) + val(seg, t) * k
        return z

    def params(self, t):
        z = self._base(t)
        for t0, amt, a, h, r in self.punches:
            if t0 <= t < t0 + a + h + r:
                if t < t0 + a:
                    z += amt * ease_out((t - t0) / a)
                elif t < t0 + a + h:
                    z += amt
                else:
                    z += amt * (1 - smooth((t - t0 - a - h) / r))
        if self.intro and t < self.intro[0]:
            z *= 1 + (self.intro[1] - 1) * (1 - ease_out(t / self.intro[0]))
        dx = dy = 0.0
        for t0, d, amp in self.shakes:
            if t0 <= t < t0 + d:
                u = (t - t0) / d
                env = (1 - u) ** 2
                dx += amp * env * math.sin(2 * math.pi * 23 * (t - t0) + 1.3)
                dy += amp * env * math.sin(2 * math.pi * 17 * (t - t0) + 0.4)
                z = max(z, 1.0 + 2.2 * amp / self.W + 0.005)
        # never reveal outside the frame
        mx, my = (z - 1) * self.W / 2, (z - 1) * self.H / 2
        return z, max(-mx, min(mx, dx)), max(-my, min(my, dy))

    def matrix(self, t):
        z, dx, dy = self.params(t)
        fx, fy = self.focus
        # keep the zoomed window inside the source: clamp the zoom centre
        fx = min(max(fx, 0), self.W)
        fy = min(max(fy, 0), self.H)
        return np.float32([[z, 0, (1 - z) * fx + dx], [0, z, (1 - z) * fy + dy]]), z

    def warp(self, img, M, z, interp=cv2.INTER_LINEAR):
        if abs(z - 1) < 1e-4 and abs(M[0, 2]) < 1e-3 and abs(M[1, 2]) < 1e-3:
            return img
        return cv2.warpAffine(img, M, (self.W, self.H), flags=interp, borderMode=cv2.BORDER_REFLECT)
