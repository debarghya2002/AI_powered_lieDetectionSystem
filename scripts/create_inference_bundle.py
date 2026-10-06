"""Package an already-trained checkpoint for video-file prediction.

This script does NOT train or fine-tune the model.  It only combines an
existing ``best.pt`` checkpoint with the matching feature-normalization values
so the Streamlit demo can safely load it for inference.

The default split reconstructs the final LOSO fold, because ``04_run_loso.py``
overwrites ``outputs/best.pt`` in every fold and leaves the final (WM027) fold
checkpoint there.  Use ``--split single_train`` only when ``best.pt`` came from
the standalone ``python train.py`` run.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.loso_splitter import generate_loso_folds
from models.deception_model import MU3DDeceptionModel
from utils.config import load_config


def _load_state(path: Path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # PyTorch < 2.0
        return torch.load(path, map_location=device)


def _feature_stats(video_ids, facial_dir, audio_dir):
    def stack(directory, suffix):
        return np.stack([np.load(Path(directory) / f"{video_id}_{suffix}.npy")
                         for video_id in video_ids])

    def mean_std(x):
        flat = x.reshape(-1, x.shape[-1])
        return flat.mean(0).astype("float32"), (flat.std(0) + 1e-6).astype("float32")

    return {
        "videomae": mean_std(stack(facial_dir, "videomae_seq")),
        "wav2vec": mean_std(stack(audio_dir, "wav2vec_seq")),
        "prosody": mean_std(stack(audio_dir, "prosody_seq")),
    }


def _select_training_ids(metadata_path, seed: int, split: str):
    df = pd.read_csv(metadata_path)
    subjects = sorted(df.subject_id.unique())
    if split == "single_train":
        shuffled = subjects.copy()
        random.Random(seed).shuffle(shuffled)
        n_val = max(1, int(0.20 * len(shuffled)))
        train_subjects = set(shuffled[n_val:])
        return df[df.subject_id.isin(train_subjects)].video_id.tolist(), "train.py split"

    folds = list(generate_loso_folds(metadata_path))
    fold_number = len(folds)  # final fold is the checkpoint that 04_run_loso leaves behind
    _, train_all, _ = folds[-1]
    by_video = df.set_index("video_id")
    train_subjects = sorted({by_video.loc[video_id, "subject_id"] for video_id in train_all})
    random.Random(seed + fold_number).shuffle(train_subjects)
    n_val = max(1, int(0.12 * len(train_subjects)))
    fit_subjects = set(train_subjects[n_val:])
    return [video_id for video_id in train_all
            if by_video.loc[video_id, "subject_id"] in fit_subjects], "final LOSO fold"


def main(config_path: str, checkpoint_path: str | None, output_path: str | None, split: str):
    cfg = load_config(config_path)
    checkpoint = Path(checkpoint_path or Path(cfg["paths"]["outputs"]) / "best.pt")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    stats_ids, origin = _select_training_ids(cfg["paths"]["metadata"], cfg.get("seed", 42), split)
    stats = _feature_stats(stats_ids, cfg["paths"]["embeddings_facial"], cfg["paths"]["embeddings_audio"])
    df = pd.read_csv(cfg["paths"]["metadata"])
    subjects = sorted(df.subject_id.unique())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    da_cfg = cfg.get("domain_adaptation", {}) or {}
    model = MU3DDeceptionModel(cfg, n_subjects=len(subjects), da_cfg=da_cfg).to(device)
    state = _load_state(checkpoint, device)
    incompatible = model.load_state_dict(state, strict=False)
    wrong_missing = [key for key in incompatible.missing_keys if not key.startswith("domain.")]
    wrong_extra = [key for key in incompatible.unexpected_keys if not key.startswith("domain.")]
    if wrong_missing or wrong_extra:
        raise ValueError(
            f"Checkpoint architecture does not match config: missing={wrong_missing}, "
            f"unexpected={wrong_extra}")

    bundle = {
        "format_version": 1,
        "model_state": model.state_dict(),
        "model_config": {"cross_modal_transformer": cfg["cross_modal_transformer"]},
        "domain_adaptation": da_cfg,
        "n_subjects": len(subjects),
        "feature_stats": {
            name: {"mean": torch.from_numpy(values[0]), "std": torch.from_numpy(values[1])}
            for name, values in stats.items()
        },
        # No calibration fitting occurs here: this is prediction-only packaging.
        "temperature": 1.0,
        "extractors": {
            "videomae_pretrained": cfg["videomae"]["pretrained"],
            "wav2vec_pretrained": cfg["wav2vec2"]["pretrained"],
            "sample_rate": cfg["audio"]["sample_rate"],
            "num_video_frames": cfg["frames"]["num_sampled"],
            "num_audio_segments": 16,
        },
        "label_semantics": {0: "lie", 1: "truth"},
        "source_checkpoint": str(checkpoint),
        "normalization_origin": origin,
    }
    destination = Path(output_path or Path(cfg["paths"]["outputs"]) / "inference_bundle.pt")
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, destination)
    print(f"Created prediction-only bundle: {destination}")
    print(f"Checkpoint: {checkpoint} | normalization: {origin} | no training performed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None, help="Default: outputs/best.pt")
    parser.add_argument("--output", default=None, help="Default: outputs/inference_bundle.pt")
    parser.add_argument("--split", choices=("last_loso", "single_train"), default="last_loso")
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.output, args.split)
