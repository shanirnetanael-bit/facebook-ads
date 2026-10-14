"""Original background music, synthesized in code (no samples, no third-party recordings -> no rights
issues). Soft "corporate / trust" bed: warm pads, a gentle piano-like arpeggio and a round bass,
with a simple algorithmic reverb. Plus ducking that lowers the music under speech.
"""
import numpy as np

SR = 48000
_rng = np.random.default_rng(11)


def midi_hz(n):
    return 440.0 * 2 ** ((n - 69) / 12)


def _adsr(n, a, d, s, r):
    e = np.full(n, s, np.float32)
    ai, di, ri = int(a * SR), int(d * SR), int(r * SR)
    ai = min(ai, n)
    e[:ai] = np.linspace(0, 1, ai)
    if di and ai < n:
        k = min(di, n - ai)
        e[ai:ai + k] = np.linspace(1, s, di)[:k]
    if ri:
        ri = min(ri, n)
        e[-ri:] *= np.linspace(1, 0, ri) ** 1.5
    return e


def pad_note(f, dur):
    n = int(dur * SR)
    t = np.arange(n) / SR
    out = np.zeros(n)
    for det in (-0.08, 0.0, 0.08):                       # 3 slightly detuned voices = chorus warmth
        ff = f * 2 ** (det / 12)
        for h, amp in ((1, 1.0), (2, 0.28), (3, 0.10), (4, 0.04)):
            out += amp * np.sin(2 * np.pi * ff * h * t + _rng.uniform(0, 6.28))
    trem = 1 + 0.06 * np.sin(2 * np.pi * 0.22 * t + _rng.uniform(0, 6.28))
    return (out * trem * _adsr(n, 0.9, 0.5, 0.75, 1.0)).astype(np.float32)


def piano_note(f, dur, vel=1.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    out = np.zeros(n)
    for h, amp, dec in ((1, 1.0, 2.2), (2, 0.45, 3.0), (3, 0.18, 4.5), (4, 0.08, 6), (5, 0.04, 8)):
        fh = f * h * (1 + 0.0004 * h * h)                # slight inharmonicity, like a real string
        out += amp * np.sin(2 * np.pi * fh * t) * np.exp(-t * dec)
    att = np.clip(t / 0.004, 0, 1)
    rel = np.ones(n)
    k = int(0.15 * SR)
    rel[-k:] = np.linspace(1, 0, k)
    return (out * att * rel * vel).astype(np.float32)


def bass_note(f, dur):
    n = int(dur * SR)
    t = np.arange(n) / SR
    out = np.sin(2 * np.pi * f * t) + 0.25 * np.sin(2 * np.pi * 2 * f * t)
    return (out * _adsr(n, 0.02, 0.4, 0.6, 0.3)).astype(np.float32)


def reverb(x, seconds=2.2, mix=0.28):
    n = int(seconds * SR)
    t = np.arange(n) / SR
    ir = _rng.standard_normal(n) * np.exp(-t * 6.9 / seconds)
    # darker tail: one-pole low-pass on the impulse response
    y = np.empty_like(ir)
    acc = 0.0
    for i in range(n):
        acc += 0.25 * (ir[i] - acc)
        y[i] = acc
    ir = y / np.sqrt((y ** 2).sum())
    L = len(x) + n
    nfft = 1 << (L - 1).bit_length()
    wet = np.fft.irfft(np.fft.rfft(x, nfft) * np.fft.rfft(ir, nfft), nfft)[:len(x)]
    return ((1 - mix) * x + mix * wet * 3).astype(np.float32)


# Imaj7 - V/vii - vi7 - IVmaj7 in D major: calm, hopeful, "you're in good hands"
PROGRESSION = [
    (50, [62, 66, 69, 73]),     # Dmaj7
    (45, [61, 64, 69, 73]),     # A/C# colour (A add9 voicing)
    (47, [62, 66, 69, 71]),     # Bm7
    (43, [62, 66, 67, 71]),     # Gmaj7
]


def compose(duration, bpm=84, fade_in=1.2, fade_out=2.0):
    beat = 60 / bpm
    bar = 4 * beat
    n = int(duration * SR) + SR
    pads = np.zeros(n, np.float32)
    keys = np.zeros(n, np.float32)
    bass = np.zeros(n, np.float32)

    def add(buf, x, t):
        i = int(t * SR)
        if i < n:
            buf[i:i + len(x)] += x[:n - i]

    t, k = 0.0, 0
    while t < duration + bar:
        root, chord = PROGRESSION[k % len(PROGRESSION)]
        for m in chord[:3]:
            add(pads, pad_note(midi_hz(m - 12), bar + 1.0), t)
        add(bass, bass_note(midi_hz(root - 12), beat * 2), t)
        add(bass, bass_note(midi_hz(root - 12), beat * 1.5), t + beat * 2.5)
        # gentle 8th-note arpeggio, one octave up, humanised timing/velocity
        pattern = [0, 2, 1, 3, 2, 1, 3, 2]
        for j, idx in enumerate(pattern):
            tj = t + j * beat / 2 + _rng.normal(0, 0.006)
            vel = (0.9 if j % 2 == 0 else 0.6) * _rng.uniform(0.85, 1.0)
            add(keys, piano_note(midi_hz(chord[idx] + 12), beat * 1.6, vel), max(tj, 0))
        t += bar
        k += 1
    mix = 0.30 * pads / (np.abs(pads).max() + 1e-9) + 0.38 * keys / (np.abs(keys).max() + 1e-9) \
        + 0.30 * bass / (np.abs(bass).max() + 1e-9)
    mix = reverb(mix)[:int(duration * SR)]
    tt = np.arange(len(mix)) / SR
    mix *= np.clip(tt / fade_in, 0, 1) * np.clip((duration - tt) / fade_out, 0, 1) ** 1.5
    return (mix / (np.abs(mix).max() + 1e-9) * 0.9).astype(np.float32)


def duck(music, voice, under_db=-22.0, open_db=-15.0, attack=0.06, release=0.45, hop=0.01):
    """Scale `music` so it sits `under_db` below the speech level while someone talks and rises to
    `open_db` below it in pauses / on the outro. Levels are relative to the speech RMS."""
    h = int(hop * SR)
    nb = len(voice) // h
    rms = np.sqrt((voice[:nb * h].reshape(nb, h) ** 2).mean(1) + 1e-12)
    db = 20 * np.log10(rms)
    speech_ref = np.percentile(db, 80)                   # typical loud-speech level
    talking = (db > speech_ref - 25).astype(np.float32)
    env = np.zeros(nb, np.float32)                       # smoothed 0..1 "voice present"
    a_up, a_dn = 1 - np.exp(-hop / attack), 1 - np.exp(-hop / release)
    acc = 0.0
    for i in range(nb):
        acc += (a_up if talking[i] > acc else a_dn) * (talking[i] - acc)
        env[i] = acc
    target_db = open_db + (under_db - open_db) * env     # per-block target, dB below speech
    m_rms = np.sqrt((music ** 2).mean() + 1e-12)
    gain_db = speech_ref + target_db - 20 * np.log10(m_rms)
    g = np.interp(np.arange(len(music)) / h, np.arange(nb), 10 ** (gain_db / 20))
    return (music * g).astype(np.float32)


# ---- "rhythmic classical" bed: baroque motor rhythm (harpsichord 16ths, cello 8ths, pizzicato, strings) ----

def harpsichord_note(f, dur, vel=1.0):
    """Bright plucked tone: many harmonics, upper ones decay faster, short quill noise at the attack."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    out = np.zeros(n)
    for h in range(1, 14):
        if f * h > 12000:
            break
        amp = (1.0 / h ** 0.8) * (1 if h % 2 else 0.7)
        out += amp * np.sin(2 * np.pi * f * h * (1 + 0.0002 * h) * t + h) * np.exp(-t * (3.0 + 0.9 * h))
    quill = _rng.standard_normal(n) * np.exp(-t * 400) * 0.25
    att = np.clip(t / 0.0015, 0, 1)
    rel = np.ones(n)
    k = min(int(0.03 * SR), n)
    rel[-k:] = np.linspace(1, 0, k)
    return ((out + quill) * att * rel * vel).astype(np.float32)


def pizz_note(f, dur=0.5, vel=1.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    out = np.zeros(n)
    for h, amp in ((1, 1.0), (2, 0.5), (3, 0.25), (4, 0.12), (5, 0.06)):
        out += amp * np.sin(2 * np.pi * f * h * t) * np.exp(-t * (9 + 4 * h))
    return (out * np.clip(t / 0.003, 0, 1) * vel).astype(np.float32)


def bowed_note(f, dur, vel=1.0, bright=8, vib=0.0):
    """Band-limited sawtooth (additive) with optional vibrato and a soft bow envelope."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    ph = 2 * np.pi * f * (t + vib * np.sin(2 * np.pi * 5.2 * t) / (2 * np.pi * 5.2))
    out = np.zeros(n)
    for h in range(1, bright + 1):
        out += np.sin(h * ph) / h * np.exp(-h / (bright * 0.6))
    return (out * _adsr(n, 0.025, 0.08, 0.75, min(0.06, dur / 3)) * vel).astype(np.float32)


def string_pad(f, dur):
    n = int(dur * SR)
    out = np.zeros(n, np.float32)
    for det in (-0.06, 0.0, 0.07):
        out += bowed_note(f * 2 ** (det / 12), dur, 0.33, bright=6, vib=0.004)
    return out * _adsr(n, 0.35, 0.3, 0.85, 0.4)


# A minor circle of fifths (Vivaldi / baroque sequence): Am - Dm - G - C - F - Dm - E - Am
BAROQUE = [
    (45, [57, 60, 64]),     # Am
    (50, [57, 62, 65]),     # Dm
    (43, [55, 59, 62]),     # G
    (48, [55, 60, 64]),     # C
    (41, [57, 60, 65]),     # F
    (38, [57, 62, 65]),     # Dm
    (40, [56, 59, 64]),     # E  (G# leading tone)
    (45, [57, 60, 64]),     # Am
]


def compose_classical(duration, bpm=112, fade_in=0.8, fade_out=2.5, intro_bars=0):
    beat = 60 / bpm
    bar = 4 * beat
    n = int(duration * SR) + 2 * SR
    hc = np.zeros(n, np.float32)
    bass = np.zeros(n, np.float32)
    pz = np.zeros(n, np.float32)
    pad = np.zeros(n, np.float32)

    def add(buf, x, t):
        i = int(max(t, 0) * SR)
        if i < n:
            buf[i:i + len(x)] += x[:n - i]

    t, k = 0.0, 0
    while t < duration + bar:
        root, chord = BAROQUE[k % len(BAROQUE)]
        sec = (k // len(BAROQUE)) % 2                      # second pass: higher, busier figure
        # harpsichord: Bach-prelude style broken chord in 16ths
        fig = [0, 1, 2, 3, 2, 1, 2, 3] if sec == 0 else [0, 2, 3, 4, 3, 2, 3, 1]
        notes = chord + [chord[0] + 12, chord[1] + 12]
        for j in range(16):
            m = notes[fig[j % 8]] + 12
            vel = (1.0 if j % 4 == 0 else 0.72) * _rng.uniform(0.9, 1.0)
            add(hc, harpsichord_note(midi_hz(m), beat * 0.9, vel), t + j * beat / 4 + _rng.normal(0, 0.003))
        # cello: driving 8ths, root / octave alternation with a passing note into the next bar
        nxt_root = BAROQUE[(k + 1) % len(BAROQUE)][0]
        line = [root, root + 12, root, root + 12, root, root + 12, root + 7, nxt_root + (1 if nxt_root < root else -1)]
        for j, m in enumerate(line):
            add(bass, bowed_note(midi_hz(m - 12 if m > 52 else m), beat * 0.42, 1.0 if j % 2 == 0 else 0.8, bright=10),
                t + j * beat / 2)
        # pizzicato violins on beats 2 and 4 (the backbeat that makes it "rhythmic")
        for b in (1, 3):
            for m in chord[1:]:
                add(pz, pizz_note(midi_hz(m + 12), 0.45, 0.8), t + b * beat + _rng.normal(0, 0.004))
        # sustained strings
        for m in chord:
            add(pad, string_pad(midi_hz(m), bar + 0.25), t)
        t += bar
        k += 1

    def norm(x):
        return x / (np.abs(x).max() + 1e-9)
    mix = 0.34 * norm(hc) + 0.30 * norm(bass) + 0.18 * norm(pz) + 0.22 * norm(pad)
    mix = reverb(mix, seconds=1.8, mix=0.22)[:int(duration * SR)]
    tt = np.arange(len(mix)) / SR
    mix *= np.clip(tt / fade_in, 0, 1) * np.clip((duration - tt) / fade_out, 0, 1) ** 1.5
    return (mix / (np.abs(mix).max() + 1e-9) * 0.9).astype(np.float32)


def carve_for_voice(x, lo_cut=90.0, dip_db=-5.0, dip_lo=1200.0, dip_hi=4500.0):
    """Leave room for speech: high-pass the bed and dip the band where speech intelligibility lives."""
    nfft = 1 << (len(x) - 1).bit_length()
    X = np.fft.rfft(x, nfft)
    f = np.fft.rfftfreq(nfft, 1 / SR)
    g = 1 / np.sqrt(1 + (lo_cut / np.maximum(f, 1)) ** 4)
    band = np.exp(-0.5 * (np.log(np.maximum(f, 1) / np.sqrt(dip_lo * dip_hi)) / np.log(dip_hi / dip_lo) * 2.4) ** 2)
    g *= 10 ** (dip_db * band / 20)
    return np.fft.irfft(X * g, nfft)[:len(x)].astype(np.float32)
