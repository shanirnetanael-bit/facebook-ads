"""Time map: source time <-> output time.

The edit is an ordered list of segments laid end to end on the output timeline:
  * Clip(src_start, src_end)        - plays a stretch of the source
  * Freeze(src_t, duration)         - holds one source frame (audio = silence)
Between two consecutive clips there is a soft cut: a short video dissolve and an equal-power audio
crossfade of `xfade` seconds, centred on the cut point. The dissolve uses "handles": material just past
the end of the outgoing clip and just before the start of the incoming one, which is the silence that
was removed anyway. Video (sources_at) and audio (audio.build_audio) read the same map, so they stay in sync.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Clip:
    src_start: float
    src_end: float
    out_start: float = 0.0
    xfade_in: float = 0.0     # crossfade with the previous clip (0 = hard cut)

    @property
    def dur(self):
        return self.src_end - self.src_start


@dataclass
class Freeze:
    src_t: float
    dur: float
    out_start: float = 0.0
    xfade_in: float = 0.0


def smoothstep(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


class TimeMap:
    def __init__(self, src_duration, xfade=0.10):
        self.src_duration = src_duration
        self.xfade = xfade
        self.segs = []

    # ---- building -------------------------------------------------------------------------
    def add_clip(self, s0, s1, soft=True):
        prev = self.segs[-1] if self.segs else None
        xf = self.xfade if (soft and isinstance(prev, Clip)) else 0.0
        self.segs.append(Clip(max(0.0, s0), min(self.src_duration, s1), xfade_in=xf))
        self._relayout()
        return self

    def add_freeze(self, src_t, dur):
        self.segs.append(Freeze(src_t, dur))
        self._relayout()
        return self

    def insert_freeze(self, src_t, dur):
        """Split the clip that contains src_t and hold that frame for `dur` seconds (e.g. a punch moment)."""
        for k, s in enumerate(self.segs):
            if isinstance(s, Clip) and s.src_start <= src_t < s.src_end:
                a = Clip(s.src_start, src_t, xfade_in=s.xfade_in)
                b = Clip(src_t, s.src_end, xfade_in=0.0)
                self.segs[k:k + 1] = [a, Freeze(src_t, dur), b]
                self._relayout()
                return self
        raise ValueError(f"src time {src_t} is not inside a kept clip")

    def _relayout(self):
        t = 0.0
        for s in self.segs:
            s.out_start = t
            t += s.dur
        self.duration = t

    @classmethod
    def from_speech(cls, speech, src_duration, keep_gap=0.14, pad_before=0.06, pad_after=0.08,
                    xfade=0.10, min_cut=0.18):
        """Cut silences. `speech` = sorted [(start, end)] of speech in source time.
        Every pause longer than keep_gap+min_cut is shortened to about keep_gap (pad_after + pad_before)."""
        tm = cls(src_duration, xfade)
        regions = []
        for s, e in speech:
            s, e = max(0.0, s - pad_before), min(src_duration, e + pad_after)
            if regions and s - regions[-1][1] < min_cut:
                regions[-1][1] = max(regions[-1][1], e)
            else:
                regions.append([s, e])
        for s, e in regions:
            tm.add_clip(s, e)
        return tm

    # ---- mapping --------------------------------------------------------------------------
    def src_to_out(self, src_t):
        """Where a source moment lands in the output. Moments in removed gaps snap to the next kept frame."""
        best = None
        for s in self.segs:
            if isinstance(s, Clip):
                if s.src_start <= src_t <= s.src_end:
                    return s.out_start + (src_t - s.src_start)
                if src_t < s.src_start and best is None:
                    best = s.out_start
        return best if best is not None else self.duration

    def _seg_index(self, t):
        for k, s in enumerate(self.segs):
            if t < s.out_start + s.dur:
                return k
        return len(self.segs) - 1

    def _src_in(self, seg, t):
        if isinstance(seg, Freeze):
            return seg.src_t
        return min(max(seg.src_start + (t - seg.out_start), 0.0), self.src_duration - 1e-3)

    def sources_at(self, t):
        """Output time -> [(source_time, weight), ...] (two entries during a dissolve)."""
        k = self._seg_index(t)
        seg = self.segs[k]
        src = self._src_in(seg, t)
        # inside the first half of a dissolve (we are in the incoming clip, blend with outgoing)
        if seg.xfade_in > 0 and t < seg.out_start + seg.xfade_in / 2 and k > 0:
            p = 0.5 + (t - seg.out_start) / seg.xfade_in
            prev = self.segs[k - 1]
            return [(self._src_in_ext(prev, t), 1 - smoothstep(p)), (src, smoothstep(p))]
        # inside the second half (we are still in the outgoing clip, blend with incoming)
        if k + 1 < len(self.segs):
            nxt = self.segs[k + 1]
            end = seg.out_start + seg.dur
            if nxt.xfade_in > 0 and t > end - nxt.xfade_in / 2:
                p = 0.5 - (end - t) / nxt.xfade_in
                return [(src, 1 - smoothstep(p)), (self._src_in_ext(nxt, t), smoothstep(p))]
        return [(src, 1.0)]

    _src_in_ext = _src_in     # _src_in already runs past clip edges (dissolve handles)

    def describe(self):
        rows = []
        for s in self.segs:
            if isinstance(s, Clip):
                rows.append(f"clip   src {s.src_start:6.2f}-{s.src_end:6.2f}  ->  out {s.out_start:6.2f}-{s.out_start + s.dur:6.2f}"
                            f"{'  (soft cut)' if s.xfade_in else ''}")
            else:
                rows.append(f"freeze src {s.src_t:6.2f} x {s.dur:.2f}s  ->  out {s.out_start:6.2f}")
        return "\n".join(rows) + f"\nduration {self.duration:.2f}s (source {self.src_duration:.2f}s)"


def detect_speech(samples, sr, floor_pct=10, above_db=15, hop_s=0.01, bridge=0.12):
    """Energy VAD -> [(start, end)] seconds. Threshold = noise floor (percentile) + above_db."""
    hop = int(sr * hop_s)
    n = len(samples) // hop
    rms = np.sqrt((samples[:n * hop].reshape(n, hop) ** 2).mean(1) + 1e-12)
    db = 20 * np.log10(rms)
    thr = np.percentile(db, floor_pct) + above_db
    active = db > thr
    out = []
    i = 0
    while i < n:
        if active[i]:
            j = i
            while j < n and active[j]:
                j += 1
            if out and i * hop_s - out[-1][1] < bridge:
                out[-1][1] = j * hop_s
            else:
                out.append([i * hop_s, j * hop_s])
            i = j
        else:
            i += 1
    return [(round(a, 3), round(b, 3)) for a, b in out if b - a > 0.05]

