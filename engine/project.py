"""Base project: wires source video + words + TimeMap + layers into frame_at(t)."""
import json
import os

from .audio import build_audio, load_audio, load_clean_voice, loudnorm_file, mix_sfx, write_wav
from .mask import MaskReader
from .render import SourceReader, blend_sources
from .timemap import TimeMap, detect_speech


class Project:
    W, H, fps = 1080, 1920, 30

    def __init__(self, source, workdir, corrections=None, clean_voice=False):
        self.source = source
        self.workdir = workdir
        raw = load_audio(source)
        self.src_duration = len(raw) / 48000
        self.speech = detect_speech(raw, 48000)          # same VAD the transcription was snapped to
        self.src_audio = load_clean_voice(source)[:len(raw)] if clean_voice else raw
        words = json.load(open(os.path.join(workdir, "words.json"), encoding="utf-8"))
        for w in words:
            w["text"] = (corrections or {}).get(w["i"], w["word"])
        self.words = words
        self.layers = []          # callables layer(frame, t_out, ctx, info) -> frame
        self.sfx = []             # (out_time, samples, gain_db)
        self.tm = None
        self.music = None         # dict(under_db=..., open_db=..., bpm=...) -> original synthesized bed, ducked
        self.camera = None        # engine.camera.Camera, applied to picture and mask alike

    # --- timeline ----------------------------------------------------------------------------
    def cut_silences(self, **kw):
        self.tm = TimeMap.from_speech(self.speech, self.src_duration, **kw)
        return self.tm

    def w(self, i, edge="start"):
        """Output time of word #i (as in the words table)."""
        return self.tm.src_to_out(self.words[i][edge])

    def words_out(self):
        return [{"i": w["i"], "text": w["text"], "start": self.w(w["i"]), "end": self.w(w["i"], "end")}
                for w in self.words if w["text"]]

    @property
    def duration(self):
        return self.tm.duration

    # --- render hooks -------------------------------------------------------------------------
    def prepare_audio(self):
        raw = os.path.join(self.workdir, "edit_audio_raw.wav")
        self.audio_path = os.path.join(self.workdir, "edit_audio.wav")
        voice = build_audio(self.tm, self.src_audio)
        track = mix_sfx(voice.copy(), self.sfx)
        if self.music is not None:
            from .music import carve_for_voice, compose, compose_classical, duck
            opts = dict(self.music)
            style, bpm = opts.pop("style", "pads"), opts.pop("bpm", None)
            if style == "classical":
                bed = carve_for_voice(compose_classical(self.tm.duration, bpm=bpm or 112))
            else:
                bed = compose(self.tm.duration, bpm=bpm or 84)
            n = min(len(bed), len(track))
            track[:n] += duck(bed[:n], voice[:n], **opts)
        write_wav(raw, track)
        loudnorm_file(raw, self.audio_path)

    def open_ctx(self):
        mask_path = os.path.join(self.workdir, "mask.mkv")
        return {"src": SourceReader(self.source),
                "mask": MaskReader(mask_path) if os.path.exists(mask_path) else None}

    def frame_at(self, t, ctx):
        sources = self.tm.sources_at(t)
        frame = blend_sources(ctx["src"], sources)
        src_t = max(sources, key=lambda s: s[1])[0]
        M = z = None
        if self.camera is not None:
            M, z = self.camera.matrix(t)
            frame = self.camera.warp(frame, M, z)
        cache = {}

        def mask():
            """Person alpha (float HxW) for this output frame, warped like the picture. Lazy + cached."""
            if "m" not in cache:
                m = ctx["mask"].get(ctx["src"].index(src_t)) if ctx["mask"] is not None else None
                if m is not None and M is not None:
                    m = self.camera.warp(m, M, z)
                cache["m"] = m
            return cache["m"]

        info = {"sources": sources, "src_t": src_t, "mask": mask}
        for layer in self.layers:
            frame = layer(frame, t, ctx, info)
        return frame
