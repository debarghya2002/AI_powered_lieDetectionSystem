"""Stage 1.1: extract audio from the (raw) video, resampled to 16kHz mono WAV.

Reads from the raw video so no audio quality is lost to video re-encoding.
"""
import subprocess
from pathlib import Path


def extract_audio(video_path: str, out_wav: str, sample_rate: int = 16000) -> str:
    Path(out_wav).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(video_path),
        "-vn",                       # no video
        "-ac", "1",                  # mono
        "-ar", str(sample_rate),     # 16kHz
        "-c:a", "pcm_s16le",
        str(out_wav),
    ]
    subprocess.run(cmd, check=True)
    return out_wav
