"""Audio: load source audio, cut it by the same TimeMap as the picture (equal-power crossfades on soft
cuts, silence on freezes), mix sound effects at output times, write a wav for the final mux."""
import os
import subprocess

import numpy as np

from .timemap import Clip, Freeze

SR = 48000


def load_audio(path, sr=SR):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).copy()


RNNOISE = os.environ.get("RNNOISE_MODEL", "/opt/vtools/models/rnnoise-sh.rnnn")


def voice_chain(denoise_mix=0.9):
    """Dialogue clean-up: rumble high-pass -> RNNoise speech denoiser -> de-mud / presence / air EQ ->
    mild de-esser -> soft compression -> soft expander (room tone between phrases down ~12 dB).
    Loudness is set later (two-pass loudnorm) on the final mix."""
    chain = ["highpass=f=75:poles=2"]
    if os.path.exists(RNNOISE):
        chain.append(f"arnndn=m={RNNOISE}:mix={denoise_mix}")
    else:
        chain.append("afftdn=nr=12:nf=-45:tn=1")
    chain += ["equalizer=f=220:t=q:w=1.0:g=-2.5",        # boxiness of a small room
              "equalizer=f=3200:t=q:w=1.2:g=3",          # presence / intelligibility
              "equalizer=f=9500:t=h:w=2000:g=1.5",       # air
              "deesser=i=0.3:m=0.5:f=0.5",
              "acompressor=threshold=0.1:ratio=2:attack=10:release=150:knee=6",
              "agate=threshold=0.004:ratio=2:range=0.25:attack=5:release=200"]
    return ",".join(chain)


def load_clean_voice(path, sr=SR):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-af", voice_chain(), "-ar", str(sr),
                          "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).copy()


def build_audio(tm, src, sr=SR, edge_fade=0.004):
    n_out = int(round(tm.duration * sr))
    out = np.zeros(n_out + sr, np.float32)
    segs = tm.segs
    for k, seg in enumerate(segs):
        o0 = int(round(seg.out_start * sr))
        if isinstance(seg, Freeze):
            continue
        xin = seg.xfade_in
        xout = segs[k + 1].xfade_in if k + 1 < len(segs) and isinstance(segs[k + 1], Clip) else 0.0
        h_in, h_out = int(xin / 2 * sr), int(xout / 2 * sr)
        s0 = int(round(seg.src_start * sr)) - h_in
        s1 = int(round(seg.src_end * sr)) + h_out
        lo = max(s0, 0)
        piece = np.zeros(s1 - s0, np.float32)
        chunk = src[lo:min(s1, len(src))]
        piece[lo - s0: lo - s0 + len(chunk)] = chunk
        g = np.ones(len(piece), np.float32)
        # equal-power fades across the dissolve window; tiny declick fades on hard edges
        nin = 2 * h_in if h_in else int(edge_fade * sr)
        nout = 2 * h_out if h_out else int(edge_fade * sr)
        if nin:
            g[:nin] *= np.sin(np.linspace(0, np.pi / 2, nin)) ** (1 if h_in else 2)
        if nout:
            g[-nout:] *= np.cos(np.linspace(0, np.pi / 2, nout)) ** (1 if h_out else 2)
        d0 = o0 - h_in
        a = max(d0, 0)
        out[a:d0 + len(piece)] += (piece * g)[a - d0:]
    return out[:n_out]


def mix_sfx(track, sfx_list, sr=SR):
    """sfx_list: [(out_time, samples_float32, gain_db)]"""
    for t, s, gain_db in sfx_list:
        i = int(t * sr)
        if i >= len(track):
            continue
        seg = s[:len(track) - i] * (10 ** (gain_db / 20))
        track[i:i + len(seg)] += seg
    return track


def write_wav(path, samples, sr=SR):
    pcm = np.clip(samples, -1, 1)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "-",
                    "-c:a", "pcm_s24le", path], input=pcm.astype(np.float32).tobytes(), check=True)


def _measure(src, target, tp, lra):
    import json
    r = subprocess.run(["ffmpeg", "-hide_banner", "-i", src, "-af",
                        f"loudnorm=I={target}:TP={tp}:LRA={lra}:print_format=json", "-f", "null", "-"],
                       capture_output=True, text=True)
    txt = r.stderr
    return json.loads(txt[txt.rindex("{"):txt.rindex("}") + 1])


def loudnorm_file(src, dst, target=-14.0, tp=-1.0, lra=11.0):
    """Loudness for Instagram/Reels (~ -14 LUFS, true peak -1 dB): gain + transparent peak limiter first
    (so the target is reachable without clipping), then a two-pass linear EBU R128 normalisation."""
    m = _measure(src, target, tp, lra)
    gain = target - float(m["input_i"])
    limited = dst + ".lim.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-af",
                    f"volume={gain + 1.5:.2f}dB,alimiter=limit=0.66:attack=3:release=60:level=disabled",
                    "-ar", str(SR), "-c:a", "pcm_s24le", limited], check=True)
    m = _measure(limited, target, tp, lra)
    af = (f"loudnorm=I={target}:TP={tp}:LRA={lra}:measured_I={m['input_i']}:measured_TP={m['input_tp']}:"
          f"measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", limited, "-af", af, "-ar", str(SR), "-c:a", "pcm_s24le", dst],
                   check=True)
    os.remove(limited)
    return dst
