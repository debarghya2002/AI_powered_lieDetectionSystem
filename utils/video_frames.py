"""Uniformly sample N RGB frames from a video file (shared by the facial streams)."""
from pathlib import Path
import cv2
import numpy as np


def sample_frames(video_path, num_frames: int = 16):
    """Return a list of `num_frames` RGB uint8 arrays, uniformly spaced in time.
    Short videos: missing positions repeat the previous frame."""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise RuntimeError(f"no frames readable in {video_path}")
    idxs = np.linspace(0, total - 1, num_frames).astype(int)
    want = set(int(i) for i in idxs)

    grabbed, i = {}, 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if i in want:
            grabbed[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        i += 1
    cap.release()

    out, last = [], None
    for idx in idxs:
        f = grabbed.get(int(idx))
        if f is None:
            f = last if last is not None else np.zeros((224, 224, 3), np.uint8)
        out.append(f)
        last = f
    return out