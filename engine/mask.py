"""Person segmentation: computes a soft alpha mask for every source frame ONCE with rembg
(u2net_human_seg) and stores it as a lossless grayscale video next to the project.

Usage:  python -m engine.mask input/source.mp4 work/mask.mkv
Read back with MaskReader(path).get(frame_index) -> float32 HxW in [0,1].
"""
import subprocess
import sys

import cv2
import numpy as np


def compute_masks(src, dst, model="u2net_human_seg", every=1):
    from rembg import new_session, remove
    from PIL import Image

    session = new_session(model)
    cap = cv2.VideoCapture(src)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ff = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{w}x{h}",
         "-r", str(fps), "-i", "-", "-c:v", "ffv1", "-level", "3", dst],
        stdin=subprocess.PIPE)
    i, last = 0, None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if last is None or i % every == 0:
            rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            m = np.asarray(remove(rgb, session=session, only_mask=True, post_process_mask=False))
            last = m
        ff.stdin.write(last.astype(np.uint8).tobytes())
        i += 1
        if i % 30 == 0:
            print(f"mask {i}/{n}", flush=True)
    ff.stdin.close()
    ff.wait()
    print("done", dst)


class MaskReader:
    """Sequential-friendly reader of the mask video (random access falls back to seeking)."""

    def __init__(self, path):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        self.pos = 0
        self.cache_idx, self.cache = -1, None

    def get(self, idx):
        if idx == self.cache_idx:
            return self.cache
        if idx != self.pos:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            self.pos = idx
        ok, fr = self.cap.read()
        if not ok:
            return self.cache if self.cache is not None else None
        self.pos += 1
        self.cache_idx = idx
        self.cache = fr[:, :, 0].astype(np.float32) / 255.0
        return self.cache


if __name__ == "__main__":
    compute_masks(sys.argv[1], sys.argv[2])
