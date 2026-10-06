"""Stage 1.1: normalize FPS + resolution and produce a standardized .mp4.

Source files carry a .wmv extension but are MP4 (ISO Media) containers, so this
re-encodes to a clean, fixed-fps, fixed-resolution H.264 mp4 (video only; audio
is handled separately by audio_extractor). Runs ffmpeg as a subprocess -> no
in-process memory growth.
"""
import subprocess
from pathlib import Path
from typing import Sequence


def normalize_video(in_path: str, out_path: str, fps: int = 15,
                    resolution: Sequence[int] = (224, 224)) -> str:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    w, h = resolution
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(in_path),
        "-r", str(fps),
        "-vf", f"scale={w}:{h}",
        "-an",                       # drop audio (extracted separately)
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)
    return out_path
