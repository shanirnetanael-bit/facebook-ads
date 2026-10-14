"""Animated word-by-word captions (reel style).

Words are grouped into short phrases (max N words / one line / break on punctuation or pauses).
A phrase sits on a rounded pill; each word pops in on its own spoken moment (fade + scale overshoot +
small rise), the word being spoken is tinted with the accent colour, and the pill glides between
widths when one phrase follows another directly.
"""
import math
import re

from .text import over, render_text, rounded_rect

PUNCT = re.compile(r"[.,!?;:\"]+$")


def ease_out_cubic(x):
    x = min(max(x, 0.0), 1.0)
    return 1 - (1 - x) ** 3


def ease_out_back(x, s=1.70158):
    x = min(max(x, 0.0), 1.0)
    return 1 + (s + 1) * (x - 1) ** 3 + s * (x - 1) ** 2


def ramp(t, t0, dur):
    return min(max((t - t0) / dur, 0.0), 1.0) if dur > 0 else float(t >= t0)


class Captions:
    def __init__(self, words, style, frame_w, frame_h):
        """words: [{'text','start','end'}] in OUTPUT time (already mapped through the TimeMap)."""
        self.s = style
        self.W, self.H = frame_w, frame_h
        self.groups = self._group(words)

    # --- layout ------------------------------------------------------------------------------
    def _word_patch(self, text, color):
        s = self.s
        return render_text(text, s["font"], s["size"], color, s.get("weight"), shadow=s["shadow"],
                           shadow_blur=s.get("shadow_blur", 6), shadow_dy=s.get("shadow_dy", 3))

    def _group(self, words):
        s = self.s
        items = []
        for w in words:
            disp = PUNCT.sub("", w["text"]).strip()
            items.append(dict(w, disp=disp, w=self._word_patch(disp, s["color"])[1]["ink_w"]))
        # 1) phrases: break on punctuation and on pauses
        phrases, cur = [], []
        for it in items:
            if cur and (PUNCT.search(cur[-1]["text"]) or it["start"] - cur[-1]["end"] > s["pause_break"]):
                phrases.append(cur)
                cur = []
            cur.append(it)
        if cur:
            phrases.append(cur)
        # 2) split each phrase into balanced chunks (no orphans: 4 -> 2+2, 5 -> 3+2) that fit one line
        groups = []
        for ph in phrases:
            n = len(ph)
            k = -(-n // s["max_words"])
            while True:
                sizes = [n // k + (1 if i < n % k else 0) for i in range(k)]
                chunks, i = [], 0
                for z in sizes:
                    chunks.append(ph[i:i + z])
                    i += z
                if all(sum(c["w"] for c in ch) + s["space"] * (len(ch) - 1) <= s["max_width"] for ch in chunks) or k >= n:
                    break
                k += 1
            groups.extend(chunks)
        # 3) a phrase that would flash by (< min_show) joins the next one if the line still fits
        merged = []
        for g in groups:
            if merged:
                prev = merged[-1]
                shown = g[0]["start"] - prev[0]["start"]
                width = sum(c["w"] for c in prev + g) + s["space"] * (len(prev) + len(g) - 1)
                if shown < s.get("min_show", 0.45) and len(prev) + len(g) <= s["max_words"] + 1 and width <= s["max_width"]:
                    merged[-1] = prev + g
                    continue
            merged.append(g)
        groups = merged
        out = []
        for g in groups:
            total = sum(c["w"] for c in g) + s["space"] * (len(g) - 1)
            x = self.W / 2 + total / 2                      # RTL: first word on the right
            for c in g:
                c["x1"] = x
                c["x0"] = x - c["w"]
                x = c["x0"] - s["space"]
            out.append({"words": g, "width": total, "start": max(0.0, g[0]["start"] - s["lead"]), "end": g[-1]["end"]})
        for a, b in zip(out, out[1:]):
            a["end"] = min(a["end"] + s["hold"], b["start"])
            b["chained"] = b["start"] - a["end"] < 0.06
        if out:
            out[-1]["end"] += s["hold"]
        return out

    # --- drawing -----------------------------------------------------------------------------
    def _rect(self, g, t):
        """Pill [left, right] at time t: right edge fixed at the first word, left edge eases out to
        each newly revealed word."""
        s = self.s
        ws = g["words"]
        R = ws[0]["x1"] + s["pad_x"]
        L = ws[0]["x0"] - s["pad_x"]
        for k in range(1, len(ws)):
            p = ease_out_cubic(ramp(t, ws[k]["start"] - s["word_lead"] - s.get("grow_lead", 0.07), s.get("grow", 0.12)))
            if p <= 0:
                break
            L += (ws[k]["x0"] - s["pad_x"] - L) * p
        return L, R

    def draw(self, frame, t):
        s = self.s
        late = []                     # outgoing words drawn on top of the gliding pill
        for gi, g in enumerate(self.groups):
            if not (g["start"] <= t < g["end"] + s["out_dur"]):
                continue
            # group envelope
            if t >= g["end"]:
                env = 1 - ramp(t, g["end"], s["out_dur"])
                if gi + 1 < len(self.groups) and self.groups[gi + 1].get("chained"):
                    # next phrase takes over the pill; old words fade out until the new first word is in
                    nxt0 = self.groups[gi + 1]["words"][0]["start"] - s["word_lead"]
                    late.append((g, 1 - ramp(t, g["end"], max(nxt0 + 0.06 - g["end"], 0.04))))
                    continue
            else:
                env = 1.0
            # pill grows right->left with the revealed words (RTL "typing"); a chained phrase glides
            # from the previous pill instead of popping in
            L, R = self._rect(g, t)
            if g.get("chained") and gi > 0:
                pL, pR = self._rect(self.groups[gi - 1], g["start"])
                p_in = ease_out_cubic(ramp(t, g["start"], s["pill_in"]))
                L, R = pL + (L - pL) * p_in, pR + (R - pR) * p_in
                pill_alpha, pill_scale = 1.0, 1.0
            else:
                p_in = ease_out_cubic(ramp(t, g["start"], s["pill_in"]))
                pill_alpha, pill_scale = p_in, 0.9 + 0.1 * p_in
            cy = s["y"]
            pw = int(round((R - L) / 2)) * 2
            pill = rounded_rect(pw, int(s["pill_h"]), s["radius"], s["pill"])
            over(frame, pill, (L + R) / 2, cy, opacity=pill_alpha * env, scale=pill_scale)
            self._draw_words(frame, g, t, env)
        for g, env in late:
            self._draw_words(frame, g, t, env)
        return frame

    def _draw_words(self, frame, g, t, env):
        s = self.s
        cy = s["y"]
        baseline = cy + s["baseline_off"]
        for k, w in enumerate(g["words"]):
            t0 = w["start"] - s["word_lead"]
            if t < t0:
                continue
            a = ramp(t, t0, s["word_fade"]) * env
            sc = 0.7 + 0.3 * ease_out_back(ramp(t, t0, s["word_pop"]))
            dy = (1 - ease_out_cubic(ramp(t, t0, s["word_pop"]))) * s["rise"]
            nxt = g["words"][k + 1]["start"] if k + 1 < len(g["words"]) else w["end"] + 0.12
            hl = min(ramp(t, w["start"] - 0.03, 0.06), 1 - ramp(t, nxt - 0.03, 0.08))
            base, meta = self._word_patch(w["disp"], s["color"])
            cx = (w["x0"] + w["x1"]) / 2
            ax = (meta["ink_x"] + meta["ink_w"] / 2) / base.shape[1]
            ay = (meta["baseline"] - s["size"] * 0.35) / base.shape[0]
            ycen = baseline - s["size"] * 0.35 + dy
            if hl < 1:
                over(frame, base, cx, ycen, opacity=a * (1 - hl) if hl > 0 else a, scale=sc, anchor=(ax, ay))
            if hl > 0:
                acc, _ = self._word_patch(w["disp"], s["accent"])
                over(frame, acc, cx, ycen, opacity=a * hl, scale=sc, anchor=(ax, ay))
