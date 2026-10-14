"""Rendering: one function frame_at(t) -> BGR frame, rendered in parallel over CPU cores,
encoded at high quality and muxed with the edited audio. Also: contact sheets for QA.

A project is any object with:
    W, H, fps, duration          output format / length (seconds)
    audio_path                   wav already cut by the TimeMap (+ sfx)
    frame_at(t, ctx) -> frame    BGR uint8 HxWx3
    open_ctx() -> ctx            per-worker resources (video readers, mask readers...)
"""
import math
import multiprocessing as mp
import os
import subprocess
import tempfile

import cv2
import numpy as np


class SourceReader:
    """Frame access by time with a sequential fast path (rendering is mostly monotonic)."""

    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.pos = 0
        self.cache = {}

    def index(self, t):
        return min(max(int(round(t * self.fps)), 0), self.n - 1)

    def get(self, idx):
        idx = min(max(idx, 0), self.n - 1)
        if idx in self.cache:
            return self.cache[idx]
        if idx < self.pos or idx > self.pos + 15:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            self.pos = idx
        frame = None
        while self.pos <= idx:
            ok, fr = self.cap.read()
            if not ok:
                break
            frame = fr
            self.pos += 1
        if frame is None:                      # past the end: reuse the last good frame
            frame = self.cache.get(max(self.cache)) if self.cache else np.zeros((1920, 1080, 3), np.uint8)
        if len(self.cache) > 8:
            self.cache.pop(min(self.cache))
        self.cache[idx] = frame
        return frame

    def at(self, t):
        return self.get(self.index(t))


def blend_sources(reader, sources):
    """[(src_t, weight)] -> BGR frame (dissolve when two sources)."""
    if len(sources) == 1:
        return reader.at(sources[0][0]).copy()
    acc = None
    for t, w in sources:
        f = reader.at(t).astype(np.float32) * w
        acc = f if acc is None else acc + f
    return np.clip(acc + 0.5, 0, 255).astype(np.uint8)


_PROJECT = None


def _render_chunk(args):
    i0, i1, path = args
    proj = _PROJECT
    ctx = proj.open_ctx()
    ff = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{proj.W}x{proj.H}",
         "-r", str(proj.fps), "-i", "-", "-c:v", "libx264", "-preset", "ultrafast", "-qp", "0",
         "-pix_fmt", "yuv444p", path], stdin=subprocess.PIPE)
    for i in range(i0, i1):
        frame = proj.frame_at(i / proj.fps, ctx)
        ff.stdin.write(np.ascontiguousarray(frame).tobytes())
    ff.stdin.close()
    ff.wait()
    return path


def render(project, out_path, workers=None, crf=16, preset="slow", fps_out=None):
    global _PROJECT
    _PROJECT = project
    workers = workers or os.cpu_count()
    n = int(math.floor(project.duration * project.fps + 1e-6))
    tmp = tempfile.mkdtemp(prefix="render_", dir=os.path.dirname(os.path.abspath(out_path)))
    bounds = np.linspace(0, n, workers * 2 + 1).astype(int)        # more chunks than cores: better balance
    jobs = [(int(a), int(b), os.path.join(tmp, f"c{k:03d}.mkv")) for k, (a, b) in enumerate(zip(bounds, bounds[1:])) if b > a]
    with mp.get_context("fork").Pool(workers) as pool:
        for k, p in enumerate(pool.imap(_render_chunk, jobs)):
            print(f"  chunk {k + 1}/{len(jobs)}", flush=True)
    lst = os.path.join(tmp, "list.txt")
    open(lst, "w").write("".join(f"file '{p}'\n" for _, _, p in jobs))
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-i", project.audio_path,
         "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
         "-profile:v", "high", "-pix_fmt", "yuv420p", "-color_primaries", "bt709", "-color_trc", "bt709",
         "-colorspace", "bt709", "-r", str(fps_out or project.fps),
         "-c:a", "aac", "-b:a", "256k", "-ar", "48000",
         "-movflags", "+faststart", "-shortest", out_path], check=True)
    for _, _, p in jobs:
        os.remove(p)
    os.remove(lst)
    os.rmdir(tmp)
    return out_path


def contact_sheet(video, out_prefix, per_sec=6, thumb_w=180, seconds_per_page=10):
    """QA sheet: `per_sec` frames for every second (one row = one second), time stamped on each."""
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = n / fps
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    th = int(thumb_w * H / W)
    times = np.arange(0, dur, 1 / per_sec)
    wanted = {min(int(round(t * fps)), n - 1): t for t in times}
    thumbs = {}
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i in wanted:
            thm = cv2.resize(fr, (thumb_w, th), interpolation=cv2.INTER_AREA)
            label = f"{wanted[i]:05.2f}s"
            cv2.rectangle(thm, (0, 0), (78, 22), (0, 0, 0), -1)
            cv2.putText(thm, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            thumbs[wanted[i]] = thm
        i += 1
    keys = sorted(thumbs)
    pages = []
    rows_per_page = seconds_per_page
    for p0 in range(0, int(math.ceil(dur)), rows_per_page):
        rows = []
        for sec in range(p0, min(p0 + rows_per_page, int(math.ceil(dur)))):
            row = [thumbs[k] for k in keys if sec <= k < sec + 1]
            row += [np.zeros_like(thumbs[keys[0]])] * (per_sec - len(row))
            rows.append(np.hstack([np.pad(r, ((2, 2), (2, 2), (0, 0))) for r in row]))
        path = f"{out_prefix}_{len(pages) + 1}.jpg"
        cv2.imwrite(path, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
        pages.append(path)
    return pages
