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
