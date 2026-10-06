"""Step 3 (audio): extract + cache the FROZEN audio SEQUENCE features per clip.

Temporal version. Per video, writes to embeddings/audio/:
    {id}_wav2vec_seq.npy   (16, 768)   wav2vec2 frame-level, binned to 16 segments
    {id}_prosody_seq.npy   (16, 7)     prosodic contour over the same 16 segments

INDEPENDENT of the facial extraction; reads .wav from Stage 1. Run once.
Resumable. First run downloads wav2vec2 weights (~360 MB).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from pathlib import Path
import numpy as np
import pandas as pd
import librosa
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **k):
        return it

import torch
from utils.config import load_config
from models.wav2vec_encoder import Wav2Vec2Encoder
from models.prosodic_features import extract_prosody_seq

N_SEGMENTS = 16


def main(config_path: str = "configs/default.yaml"):
    cfg = load_config(config_path)
    df = pd.read_csv(cfg["paths"]["metadata"])
    out_dir = Path(cfg["paths"]["embeddings_audio"])
    out_dir.mkdir(parents=True, exist_ok=True)
    sr = cfg["audio"]["sample_rate"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  videos={len(df)}  sr={sr}  segments={N_SEGMENTS}")

    encoder = Wav2Vec2Encoder(cfg["wav2vec2"]["pretrained"], device=device,
                              sr=sr, n_segments=N_SEGMENTS)

    done = skip = fail = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Stage 3 audio (temporal)"):
        vid = row["video_id"]
        w_out = out_dir / f"{vid}_wav2vec_seq.npy"
        p_out = out_dir / f"{vid}_prosody_seq.npy"
        if w_out.exists() and p_out.exists():
            skip += 1
            continue
        apath = Path(row["audio_path"])
        if not apath.exists():
            fail += 1
            continue
        try:
            wav, _ = librosa.load(str(apath), sr=sr, mono=True)
            wav = wav.astype("float32")
            np.save(w_out, encoder(wav))
            np.save(p_out, extract_prosody_seq(wav, sr=sr, n_segments=N_SEGMENTS))
            done += 1
        except Exception as e:  # noqa: BLE001
            fail += 1
            print(f"FAILED {vid}: {e}", file=sys.stderr)

    print(f"audio extraction: done={done} skipped={skip} failed={fail}")


if __name__ == "__main__":
    main()