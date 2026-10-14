"""Small synthesized sound effects (no external assets): whoosh, pop, thud, shimmer, tick."""
import numpy as np

SR = 48000
_rng = np.random.default_rng(7)


def _env(n, attack, release):
    e = np.ones(n, np.float32)
    a, r = int(attack * SR), int(release * SR)
    if a:
        e[:a] = np.linspace(0, 1, a) ** 2
    if r:
        e[-r:] *= np.linspace(1, 0, r) ** 2
    return e


def _onepole(x, cut):
    """Time-varying one-pole low-pass; cut = array of cutoff Hz per sample."""
    y = np.empty_like(x)
    a = 1 - np.exp(-2 * np.pi * cut / SR)
    acc = 0.0
    for i in range(len(x)):
        acc += a[i] * (x[i] - acc)
        y[i] = acc
    return y


def whoosh(dur=0.45, lo=300, hi=4500, peak=0.55):
    n = int(dur * SR)
    noise = _rng.standard_normal(n).astype(np.float32)
    u = np.linspace(0, 1, n)
    sweep = lo + (hi - lo) * np.sin(np.pi * np.clip(u / peak, 0, 1) / 2) * (1 - np.clip((u - peak) / (1 - peak), 0, 1)) ** 1.5
    lp = _onepole(noise, sweep)
    bp = lp - _onepole(lp, sweep * 0.25)
    env = np.sin(np.pi * u) ** 2 * np.where(u < peak, (u / peak) ** 0.5, 1)
    out = bp * env
    return (out / (np.abs(out).max() + 1e-9) * 0.8).astype(np.float32)


def pop(f0=900, f1=420, dur=0.08):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = f1 + (f0 - f1) * np.exp(-t * 60)
    ph = 2 * np.pi * np.cumsum(f) / SR
    out = np.sin(ph) * np.exp(-t * 45)
    out[:int(0.002 * SR)] *= np.linspace(0, 1, int(0.002 * SR))
    return (out * 0.8).astype(np.float32)


def thud(dur=0.45):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = 52 + 80 * np.exp(-t * 30)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 9)
    click = _onepole(_rng.standard_normal(n).astype(np.float32), np.full(n, 1800.0)) * np.exp(-t * 70) * 2.5
    out = body + click
    return (out / np.abs(out).max() * 0.9).astype(np.float32)


def shimmer(dur=1.1):
    n = int(dur * SR)
    t = np.arange(n) / SR
    out = np.zeros(n)
    for k, (f, a, d) in enumerate([(1760, 1.0, 4.5), (2637, 0.6, 5.5), (3520, 0.35, 7), (5274, 0.2, 9)]):
        out += a * np.sin(2 * np.pi * f * t + 0.3 * np.sin(2 * np.pi * 6 * t)) * np.exp(-t * d) * \
            np.clip((t - 0.025 * k) * 400, 0, 1)
    return (out / np.abs(out).max() * 0.6).astype(np.float32)


def tick():
    return pop(1600, 1100, 0.05)
