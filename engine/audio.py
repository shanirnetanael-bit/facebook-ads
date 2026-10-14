"""Audio: load source audio, cut it by the same TimeMap as the picture (equal-power crossfades on soft
cuts, silence on freezes), mix sound effects at output times, write a wav for the final mux."""
import subprocess

import numpy as np

from .timemap import Clip, Freeze

SR = 48000


def load_audio(path, sr=SR):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
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


def loudnorm_file(src, dst, target=-14.0, tp=-1.0, lra=11.0):
    """Two-pass EBU R128 loudness normalisation (Instagram/Reels ~ -14 LUFS)."""
    import json
    r = subprocess.run(["ffmpeg", "-hide_banner", "-i", src, "-af",
                        f"loudnorm=I={target}:TP={tp}:LRA={lra}:print_format=json", "-f", "null", "-"],
                       capture_output=True, text=True)
    txt = r.stderr
    m = json.loads(txt[txt.rindex("{"):txt.rindex("}") + 1])
    af = (f"loudnorm=I={target}:TP={tp}:LRA={lra}:measured_I={m['input_i']}:measured_TP={m['input_tp']}:"
          f"measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-af", af, "-ar", str(SR), "-c:a", "pcm_s24le", dst],
                   check=True)
    return dst
