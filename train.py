"""Train the multi-task (+ optional domain-adversarial) TEMPORAL model on one
subject-stratified train/val split. Sanity check; full LOSO is Stage 9.

Run:  python train.py
Toggle domain adaptation via configs/train.yaml -> domain_adaptation.enabled.
"""
import argparse
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from utils.config import load_config
from utils.seed import set_seed
from utils.device import get_device
from datasets.mu3d_dataset import MU3DDataset
from models.deception_model import MU3DDeceptionModel
from multitask.multitask_loss import MultiTaskLoss
from trainers.multitask_trainer import MultiTaskTrainer


def feature_stats(ids, facial_dir, audio_dir):
    """Per-channel mean/std over the train split for each cached SEQUENCE feature."""
    def stack(d, suffix):
        return np.stack([np.load(Path(d) / f"{v}_{suffix}.npy") for v in ids])  # (N,T,D)
    ms = lambda a: (a.reshape(-1, a.shape[-1]).mean(0).astype("float32"),
                    (a.reshape(-1, a.shape[-1]).std(0) + 1e-6).astype("float32"))
    return {"videomae": ms(stack(facial_dir, "videomae_seq")),
            "wav2vec":  ms(stack(audio_dir, "wav2vec_seq")),
            "prosody":  ms(stack(audio_dir, "prosody_seq"))}


def main(config_path: str = "configs/default.yaml"):
    cfg = load_config(config_path)
    set_seed(cfg.get("seed", 42))
    device = get_device(cfg.get("device", "auto"))

    df = pd.read_csv(cfg["paths"]["metadata"])
    subjects = sorted(df.subject_id.unique())
    random.Random(cfg.get("seed", 42)).shuffle(subjects)
    n_val = max(1, int(0.2 * len(subjects)))
    val_subj, train_subj = set(subjects[:n_val]), set(subjects[n_val:])

    train_ids = df[df.subject_id.isin(train_subj)].video_id.tolist()
    val_ids = df[df.subject_id.isin(val_subj)].video_id.tolist()

    tr_anx = df[df.subject_id.isin(train_subj)].anxiety
    a_mean, a_std = float(tr_anx.mean()), float(tr_anx.std())
    feat_stats = feature_stats(train_ids, cfg["paths"]["embeddings_facial"], cfg["paths"]["embeddings_audio"])
    subj_to_idx = {s: i for i, s in enumerate(subjects)}

    def make(ids):
        return MU3DDataset(cfg["paths"]["metadata"], ids,
                           cfg["paths"]["embeddings_facial"], cfg["paths"]["embeddings_audio"],
                           a_mean, a_std, feat_stats, subj_to_idx)

    bs, nw = cfg["runtime"]["batch_size"], cfg["runtime"]["num_workers"]
    train_loader = DataLoader(make(train_ids), batch_size=bs, shuffle=True,
                              num_workers=nw, drop_last=True)
    val_loader = DataLoader(make(val_ids), batch_size=bs, shuffle=False, num_workers=nw)

    da = cfg.get("domain_adaptation", {}) or {}
    model = MU3DDeceptionModel(cfg, n_subjects=len(subjects), da_cfg=da)
    grl_lambda = float(da.get("grl_lambda", 0.0)) if da.get("enabled") else 0.0

    weights = {"lie": cfg["multitask"]["lie_weight"],
               "anxiety": cfg["multitask"]["anxiety_weight"]}
    loss_fn = MultiTaskLoss(weights)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["optim"]["lr"],
                            weight_decay=cfg["optim"]["weight_decay"])
    trainer = MultiTaskTrainer(model, loss_fn, opt, device, cfg, ckpt_dir=cfg["paths"]["outputs"])

    da_on = model.domain is not None and grl_lambda > 0
    print(f"train clips={len(train_ids)}  val clips={len(val_ids)}  device={device}  "
          f"domain_adaptation={'ON (lambda=%.2f)' % grl_lambda if da_on else 'off'}")
    trainer.train(train_loader, val_loader,
                  epochs=cfg["optim"]["epochs"],
                  grad_accum=cfg["runtime"]["grad_accum_steps"],
                  patience=cfg["optim"]["early_stopping_patience"],
                  grl_lambda=grl_lambda)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    main(p.parse_args().config)
