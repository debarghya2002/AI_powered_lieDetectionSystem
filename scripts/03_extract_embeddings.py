"""Step 3 (facial): extract + cache the FROZEN facial features for every clip.

Per video, writes to embeddings/facial/:
    {id}_videomae_seq.npy  (8, 768)    VideoMAE temporal token sequence (frozen)
    {id}_landmarks.npy     (T, 1404)   normalized MediaPipe landmark sequence

Temporal version of Stage 2 (no pooling over time). Run once. Resumable: skips
files already cached, so existing landmark arrays are reused and only the new
videomae sequences are computed. First run downloads VideoMAE weights (~300 MB).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from pathlib import Path
import numpy as np
import pandas as pd
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **k):
        return it

import torch
from utils.config import load_config
from utils.video_frames import sample_frames
from models.videomae_encoder import VideoMAEEncoder
from models.landmark_extractor import extract_landmarks


def main(config_path: str = "configs/default.yaml"):
    cfg = load_config(config_path)
    df = pd.read_csv(cfg["paths"]["metadata"])
    out_dir = Path(cfg["paths"]["embeddings_facial"])
    out_dir.mkdir(parents=True, exist_ok=True)
    n_frames = cfg["frames"]["num_sampled"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  videos={len(df)}  frames/clip={n_frames}")

    encoder = VideoMAEEncoder(cfg["videomae"]["pretrained"], device=device)

    done = skip = fail = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Stage 2 facial (temporal)"):
        vid = row["video_id"]
        v_out = out_dir / f"{vid}_videomae_seq.npy"
        l_out = out_dir / f"{vid}_landmarks.npy"
        if v_out.exists() and l_out.exists():
            skip += 1
            continue
        vpath = Path(row["video_path"])
        if not vpath.exists():
            fail += 1
            continue
        try:
            frames = sample_frames(vpath, num_frames=n_frames)
            if not v_out.exists():
                np.save(v_out, encoder(frames))
            if not l_out.exists():
                np.save(l_out, extract_landmarks(frames))
            done += 1
        except Exception as e:  # noqa: BLE001
            fail += 1
            print(f"FAILED {vid}: {e}", file=sys.stderr)

    print(f"facial extraction: done={done} skipped={skip} failed={fail}")


if __name__ == "__main__":
    main()