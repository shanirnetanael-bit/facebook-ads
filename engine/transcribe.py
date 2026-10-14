"""Word-level transcription.

Two backends:
  * whisper.cpp + ivrit.ai Hebrew model (preferred; set WHISPER_MODEL) - used when the ggml file exists.
    Steps: 16 kHz wav -> whisper-cli full JSON (per-token offsets) -> merge tokens into words -> refine().
  * engine.asr: local ONNX Whisper large-v3-turbo with cross-attention/DTW word times (fallback, e.g. when
    huggingface.co is not reachable to fetch the ivrit.ai model) -> snap_to_speech().

Usage:  python -m engine.transcribe input/source.mp4 work/ ["context prompt: names, domain terms"]
        (if work/transcript_raw.json or work/asr_raw.json already exists, only re-builds words.json / words.md)
"""
import json
import os
import subprocess
import sys

WHISPER_BIN = os.environ.get("WHISPER_BIN", "/opt/vtools/whisper.cpp/build/bin/whisper-cli")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "/opt/vtools/models/ggml-ivrit-large-v3-turbo.bin")


def run_whisper(video, workdir, lang="he", threads=os.cpu_count()):
    wav = os.path.join(workdir, "audio16k.wav")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", video, "-ar", "16000", "-ac", "1",
                    "-c:a", "pcm_s16le", wav], check=True)
    subprocess.run([WHISPER_BIN, "-m", WHISPER_MODEL, "-f", wav, "-l", lang, "-t", str(threads),
                    "-ojf", "-of", os.path.join(workdir, "transcript_raw"), "-np"], check=True)


def tokens_to_words(raw):
    """whisper tokens -> [{'i','word','start','end','p'}]; a token starting with a space opens a new word."""
    words = []
    for si, seg in enumerate(raw["transcription"]):
        seg_span = (seg["offsets"]["from"] / 1000, seg["offsets"]["to"] / 1000)
        for tok in seg["tokens"]:
            text = tok["text"]
            if text.startswith("[_") or text.startswith("<|"):
                continue
            t0, t1 = tok["offsets"]["from"] / 1000, tok["offsets"]["to"] / 1000
            if text.startswith(" ") or not words:
                words.append({"word": text.strip(), "start": t0, "end": t1, "ps": [tok["p"]], "seg": si, "seg_span": seg_span})
            else:
                words[-1]["word"] += text
                words[-1]["end"] = max(words[-1]["end"], t1)
                words[-1]["ps"].append(tok["p"])
    out = []
    for w in words:
        if not w["word"].strip(".,!?- "):        # bare punctuation token -> glue to previous word
            if out:
                out[-1]["word"] += w["word"]
            continue
        out.append({"i": len(out), "word": w["word"], "start": round(w["start"], 2),
                    "end": round(max(w["end"], w["start"] + 0.05), 2), "p": round(min(w["ps"]), 2),
                    "seg": w["seg"], "seg_span": w["seg_span"]})
    return out


def refine(words, speech, min_dur=0.12):
    """Improve whisper's rough word times using the audio:
    1. each whisper segment is stretched onto the real speech found inside it (energy VAD),
    2. runs of collapsed words (near-zero duration, typical at segment starts) are merged with the next
       word and the span is re-divided in proportion to letter count,
    3. starts are forced monotonic."""
    by_seg = {}
    for w in words:
        by_seg.setdefault(w["seg"], []).append(w)
    for ws in by_seg.values():
        s0, s1 = ws[0]["seg_span"]
        # snap a segment edge to the audio only where the audio really has an edge (not mid-speech)
        on = [a for a, b in speech if s0 - 0.25 <= a <= s1]
        off = [b for a, b in speech if s0 <= b <= s1 + 0.25]
        ongoing_s0 = any(a < s0 - 0.25 and b > s0 for a, b in speech)
        ongoing_s1 = any(a < s1 and b > s1 + 0.25 for a, b in speech)
        t0 = s0 if ongoing_s0 or not on else on[0]
        t1 = s1 if ongoing_s1 or not off else off[-1]
        if t1 - t0 > 0.2:
            k = (t1 - t0) / max(s1 - s0, 1e-3)
            for w in ws:
                w["start"] = t0 + (w["start"] - s0) * k
                w["end"] = t0 + (w["end"] - s0) * k
        i = 0
        while i < len(ws):
            j = i
            while j < len(ws) and ws[j]["end"] - ws[j]["start"] < min_dur:
                j += 1
            if j > i:
                run = ws[i:min(j + 1, len(ws))]
                a, b = run[0]["start"], max(run[-1]["end"], run[0]["start"] + min_dur * len(run))
                n = [len(w["word"].strip(".,!?")) + 1 for w in run]
                c = a
                for w, m in zip(run, n):
                    w["start"], w["end"] = c, c + (b - a) * m / sum(n)
                    c = w["end"]
                i = j + 1
            else:
                i += 1
    for prev, w in zip(words, words[1:]):
        w["start"] = max(w["start"], prev["start"] + 0.05)
        w["end"] = max(w["end"], w["start"] + 0.05)
    for w in words:
        w["start"], w["end"] = round(w["start"], 2), round(w["end"], 2)
    return words


def snap_to_speech(words, speech, max_move=1.0):
    """Pull word edges that fall in silence onto the nearest speech edge (DTW times are ~20-40 ms granular
    but a word can swallow a leading/trailing pause)."""
    def inside(t):
        return any(a <= t <= b for a, b in speech)
    for w in words:
        if not inside(w["start"]):
            nxt = [a for a, b in speech if w["start"] < a < w["end"]]
            if nxt and nxt[0] - w["start"] <= max_move:
                w["start"] = nxt[0]
        if not inside(w["end"]):
            prv = [b for a, b in speech if w["start"] < b < w["end"]]
            if prv and w["end"] - prv[-1] <= max_move:
                w["end"] = prv[-1]
    for prev, w in zip(words, words[1:]):
        w["start"] = max(w["start"], prev["start"] + 0.05)
        w["end"] = max(w["end"], w["start"] + 0.08)
    for i, w in enumerate(words):
        w["i"] = i
        w["start"], w["end"], w["p"] = round(w["start"], 2), round(w["end"], 2), round(w["p"], 2)
    return words


def write_table(words, path):
    lines = ["| # | מילה | התחלה (שנ') | סוף (שנ') |", "|---|---|---|---|"]
    for w in words:
        lines.append(f"| {w['i']} | {w['word']} | {w['start']:.2f} | {w['end']:.2f} |")
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")


def build(workdir, speech=None):
    onnx_raw = os.path.join(workdir, "asr_raw.json")
    if os.path.exists(onnx_raw):
        words = snap_to_speech(json.load(open(onnx_raw, encoding="utf-8"))["words"], speech or [])
    else:
        raw = json.load(open(os.path.join(workdir, "transcript_raw.json"), encoding="utf-8"))
        words = tokens_to_words(raw)
        if speech is not None:
            words = refine(words, speech)
    json.dump(words, open(os.path.join(workdir, "words.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    write_table(words, os.path.join(workdir, "words.md"))
    return words


if __name__ == "__main__":
    video, workdir = sys.argv[1], sys.argv[2]
    prompt = sys.argv[3] if len(sys.argv) > 3 else None
    from .audio import load_audio
    from .timemap import detect_speech
    speech = detect_speech(load_audio(video), 48000)
    have_raw = any(os.path.exists(os.path.join(workdir, f)) for f in ("transcript_raw.json", "asr_raw.json"))
    if not have_raw and os.path.exists(WHISPER_MODEL):
        run_whisper(video, workdir)
    elif not have_raw:
        from . import asr
        ws = asr.transcribe(load_audio(video, 16000), speech, prompt=prompt)
        json.dump({"words": ws}, open(os.path.join(workdir, "asr_raw.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
    for w in build(workdir, speech):
        print(w["i"], w["start"], w["end"], w["p"], w["word"])
