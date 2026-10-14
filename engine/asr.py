"""Local Whisper (ONNX) with word timestamps from cross-attention + DTW, no internet needed at run time.

Used when the whisper.cpp / ivrit.ai ggml model is not available (e.g. huggingface.co blocked). Model:
sherpa-onnx's export of whisper large-v3-turbo (int8), downloaded from GitHub releases by scripts/setup.sh.

How word times are found (same method as openai-whisper's word_timestamps):
  1. the decoder graph is patched once so its cross-attention softmax maps become extra outputs,
  2. greedy decoding gives the text tokens,
  3. one forced pass over [sot.. text.. eot] returns attention (tokens x audio frames, 20 ms each),
  4. heads of the upper decoder layers are normalised, median-filtered, averaged and aligned with DTW,
  5. token times are grouped into words (a token starting with a space opens a new word).
"""
import base64
import os

import numpy as np
import onnx
import onnxruntime as ort

MODEL_DIR = os.environ.get("WHISPER_ONNX_DIR", "/opt/vtools/models/sherpa-onnx-whisper-turbo")
SR = 16000
N_FRAMES = 3000                    # 30 s of mel frames
CROSS_SOFTMAX = ["/Softmax_1_output_0", "/Softmax_3_output_0", "/Softmax_5_output_0", "/Softmax_7_output_0"]


def _patched_decoder(src, dst):
    if os.path.exists(dst):
        return dst
    m = onnx.load(src)
    for name in CROSS_SOFTMAX:
        m.graph.output.append(onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None))
    onnx.save(m, dst)
    return dst


def log_mel(audio, n_mels=128):
    import librosa
    audio = np.concatenate([audio.astype(np.float32), np.zeros(SR * 30, np.float32)])
    win = np.hanning(401)[:-1].astype(np.float32)                       # periodic hann, as torch.hann_window
    pad = np.pad(audio, (200, 200), mode="reflect")
    n = 1 + (len(pad) - 400) // 160
    frames = np.lib.stride_tricks.as_strided(pad, (n, 400), (pad.strides[0] * 160, pad.strides[0]))
    spec = np.abs(np.fft.rfft(frames * win, axis=1)) ** 2
    spec = spec[:-1].T                                                  # drop last frame like whisper
    fb = librosa.filters.mel(sr=SR, n_fft=400, n_mels=n_mels)
    mel = np.log10(np.maximum(fb @ spec, 1e-10))
    mel = np.maximum(mel, mel.max() - 8.0)
    mel = (mel + 4.0) / 4.0
    return mel[:, :N_FRAMES][None].astype(np.float32)


def _dtw(cost):
    N, M = cost.shape
    D = np.full((N + 1, M + 1), np.inf)
    D[0, 0] = 0
    T = np.zeros((N + 1, M + 1), np.int8)
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            c = (D[i - 1, j - 1], D[i - 1, j], D[i, j - 1])
            k = int(np.argmin(c))
            D[i, j] = cost[i - 1, j - 1] + c[k]
            T[i, j] = k
    i, j, path = N, M, []
    while i > 0 or j > 0:
        path.append((i - 1, j - 1))
        k = T[i, j] if i > 0 and j > 0 else (1 if i > 0 else 2)
        if k == 0:
            i, j = i - 1, j - 1
        elif k == 1:
            i -= 1
        else:
            j -= 1
    path = np.array(path[::-1])
    return path[:, 0], path[:, 1]


class Whisper:
    def __init__(self, model_dir=MODEL_DIR, threads=os.cpu_count()):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        prov = ["CPUExecutionProvider"]
        self.enc = ort.InferenceSession(os.path.join(model_dir, "turbo-encoder.int8.onnx"), so, providers=prov)
        dec = _patched_decoder(os.path.join(model_dir, "turbo-decoder.int8.onnx"),
                               os.path.join(model_dir, "turbo-decoder.int8.attn.onnx"))
        self.dec = ort.InferenceSession(dec, so, providers=prov)
        meta = self.enc.get_modelmeta().custom_metadata_map
        self.meta = meta
        self.n_layer = int(meta["n_text_layer"])
        self.n_state = int(meta["n_text_state"])
        self.sot, self.eot = int(meta["sot"]), int(meta["eot"])
        self.transcribe_tok, self.no_ts = int(meta["transcribe"]), int(meta["no_timestamps"])
        codes = meta["all_language_codes"].split(",")
        toks = [int(x) for x in meta["all_language_tokens"].split(",")]
        self.lang_tok = dict(zip(codes, toks))
        self.vocab = {}
        for line in open(os.path.join(model_dir, "turbo-tokens.txt"), encoding="utf-8"):
            parts = line.split()
            if len(parts) == 2:
                self.vocab[int(parts[1])] = base64.b64decode(parts[0])

    def _caches(self):
        z = np.zeros((self.n_layer, 1, 448, self.n_state), np.float32)
        return z, z.copy()

    def _decode(self, tokens, k, v, ck, cv, offset):
        out = self.dec.run(None, {"tokens": np.array([tokens], np.int64), "in_n_layer_self_k_cache": k,
                                  "in_n_layer_self_v_cache": v, "n_layer_cross_k": ck, "n_layer_cross_v": cv,
                                  "offset": np.array([offset], np.int64)})
        return out

    def encode_text(self, text):
        """Greedy longest-match tokenisation (good enough for a context prompt)."""
        if not hasattr(self, "_rev"):
            self._rev = {b: i for i, b in self.vocab.items() if i < self.eot}
        data, toks, i = (" " + text).encode("utf-8"), [], 0
        while i < len(data):
            for j in range(min(len(data), i + 32), i, -1):
                if data[i:j] in self._rev:
                    toks.append(self._rev[data[i:j]])
                    i = j
                    break
            else:
                i += 1
        return toks

    def transcribe_chunk(self, audio, lang="he", max_tokens=220, prompt=None):
        """audio: float32 16 kHz <= 30 s. Returns [{'word','start','end','p'}] with times relative to chunk.
        prompt: optional context text (names, domain terms) that steers spelling."""
        ck, cv = self.enc.run(None, {"mel": log_mel(audio)})
        sot_seq = [self.sot, self.lang_tok[lang], self.transcribe_tok, self.no_ts]
        if prompt:
            sot_seq = [int(self.meta["sot_prev"])] + self.encode_text(prompt)[-200:] + sot_seq
        k, v = self._caches()
        logits, k, v = self._decode(sot_seq, k, v, ck, cv, 0)[:3]
        text, probs = [], []
        offset = len(sot_seq)
        suppress_blank = True
        for _ in range(max_tokens):
            lg = logits[0, -1].astype(np.float64)
            lg[self.eot + 1:] = -np.inf                                   # no special/timestamp tokens
            if suppress_blank:
                lg[220] = -np.inf
                lg[self.eot] = -np.inf
                suppress_blank = False
            tok = int(np.argmax(lg))
            pr = np.exp(lg - lg.max())
            p = float(pr[tok] / pr.sum())
            if tok == self.eot:
                break
            text.append(tok)
            probs.append(p)
            logits, k, v = self._decode([tok], k, v, ck, cv, offset)[:3]
            offset += 1
            if len(text) > 8 and text[-4:] == text[-8:-4]:            # repetition loop guard
                break
        if not text:
            return []
        # forced pass with attention
        k, v = self._caches()
        seq = sot_seq + text + [self.eot]
        out = self._decode(seq, k, v, ck, cv, 0)
        heads = np.concatenate([a[0] for a in out[3 + self.n_layer // 2:]], 0)   # upper-half layers: (heads, tok, 1500)
        n_frames = min(1500, int(np.ceil(len(audio) / SR * 50)))
        w = heads[:, :, :n_frames].astype(np.float64)
        w = (w - w.mean(-2, keepdims=True)) / (w.std(-2, keepdims=True) + 1e-9)
        from scipy.ndimage import median_filter
        w = median_filter(w, size=(1, 1, 7))
        matrix = w.mean(0)[len(sot_seq) - 1:-1]                          # rows: text tokens (+ eot)
        ti, tj = _dtw(-matrix)
        jumps = np.pad(np.diff(ti), (1, 0), constant_values=1).astype(bool)
        jump_times = tj[jumps] / 50.0                                    # time at which each token starts
        # group tokens into words
        words, cur = [], None
        for idx, (tok, p) in enumerate(zip(text, probs)):
            b = self.vocab.get(tok, b"")
            if cur is None or b.startswith(b" "):
                cur = {"bytes": b, "t": [idx], "p": [p]}
                words.append(cur)
            else:
                cur["bytes"] += b
                cur["t"].append(idx)
                cur["p"].append(p)
        res = []
        for wd in words:
            s = jump_times[wd["t"][0]] if wd["t"][0] < len(jump_times) else jump_times[-1]
            e_idx = wd["t"][-1] + 1
            e = jump_times[e_idx] if e_idx < len(jump_times) else n_frames / 50
            res.append({"word": wd["bytes"].decode("utf-8", "replace").strip(), "start": float(s),
                        "end": float(max(e, s + 0.04)), "p": float(min(wd["p"]))})
        return res


def split_chunks(speech, total, max_len=24.0):
    """Cut points in the longest pauses so that every chunk is <= max_len seconds."""
    gaps = [((a1 + b0) / 2, b0 - a1) for (a0, a1), (b0, b1) in zip(speech, speech[1:])]
    cuts = [0.0]
    while total - cuts[-1] > max_len:
        cand = [g for g in gaps if cuts[-1] + 4 < g[0] <= cuts[-1] + max_len]
        nxt = max(cand, key=lambda g: g[1])[0] if cand else cuts[-1] + max_len
        cuts.append(nxt)
    cuts.append(total)
    return list(zip(cuts, cuts[1:]))


def transcribe(audio16k, speech, lang="he", prompt=None):
    model = Whisper()
    words = []
    total = len(audio16k) / SR
    for c0, c1 in split_chunks(speech, total):
        seg = audio16k[int(c0 * SR):int(c1 * SR)]
        ws = model.transcribe_chunk(seg, lang, prompt=prompt)
        print(f"  chunk {c0:6.2f}-{c1:6.2f}: {' '.join(w['word'] for w in ws)}", flush=True)
        for w in ws:
            w["start"] += c0
            w["end"] += c0
        words += ws
    return words
