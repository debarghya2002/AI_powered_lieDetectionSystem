"""Stage 9.4-9.5: Leave-One-Subject-Out evaluation.

For each subject: train on the other subjects (with a small inner validation
split for early stopping + temperature scaling), calibrate, then predict the
held-out subject's clips. All held-out predictions are pooled, saved, and
scored once -> trustworthy LOSO metrics. Respects the domain-adaptation config.

grl_lambda_override: if given, overrides the grl_lambda strength used for THIS
run (the domain-adaptation branch must still be enabled in the config for this
to have any effect -- the override changes strength, not on/off). Lets you
sweep lambda values without editing the YAML each time.

out_tag: if given, predictions/metrics are written to loso_predictions_<tag>.csv
/ loso_metrics_<tag>.json instead of the default filenames, so experiments never
overwrite your canonical saved results.
"""
import json
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
from datasets.loso_splitter import generate_loso_folds
from models.deception_model import MU3DDeceptionModel
from multitask.multitask_loss import MultiTaskLoss
from trainers.multitask_trainer import MultiTaskTrainer
from calibration.temperature_scaling import TemperatureScaler
from evaluation.metrics import compute_metrics


def _feature_stats(ids, fdir, adir):
    def stack(d, suf):
        return np.stack([np.load(Path(d) / f"{v}_{suf}.npy") for v in ids])  # (N,T,D)
    ms = lambda a: (a.reshape(-1, a.shape[-1]).mean(0).astype("float32"),
                    (a.reshape(-1, a.shape[-1]).std(0) + 1e-6).astype("float32"))
    return {"videomae": ms(stack(fdir, "videomae_seq")),
            "wav2vec":  ms(stack(adir, "wav2vec_seq")),
            "prosody":  ms(stack(adir, "prosody_seq"))}


@torch.no_grad()
def _logits(model, loader, device):
    model.eval()
    L, Y = [], []
    for b in loader:
        b = {k: v.to(device) for k, v in b.items()}
        out, _ = model(b["videomae"], b["landmarks"], b["wav2vec"], b["prosody"])
        L.append(out["lie"]); Y.append(b["label"])
    return torch.cat(L), torch.cat(Y)


def run_loso(config_path: str = "configs/default.yaml", n_folds=None,
             grl_lambda_override=None, out_tag=None):
    cfg = load_config(config_path)
    set_seed(cfg.get("seed", 42))
    device = get_device(cfg.get("device", "auto"))
    meta = cfg["paths"]["metadata"]
    fdir, adir = cfg["paths"]["embeddings_facial"], cfg["paths"]["embeddings_audio"]

    df = pd.read_csv(meta)
    by_vid = df.set_index("video_id")
    subjects = sorted(df.subject_id.unique())
    subj_to_idx = {s: i for i, s in enumerate(subjects)}
    da = cfg.get("domain_adaptation", {}) or {}
    grl_lambda = float(da.get("grl_lambda", 0.0)) if da.get("enabled") else 0.0
    if grl_lambda_override is not None:
        grl_lambda = float(grl_lambda_override)

    folds = list(generate_loso_folds(meta))
    if n_folds:
        folds = folds[:n_folds]
    tag_note = f"  [tag={out_tag}]" if out_tag else ""
    print(f"LOSO: {len(folds)} folds | device={device} | "
          f"domain_adaptation={'ON' if grl_lambda > 0 else 'off'} (lambda={grl_lambda:.2f}){tag_note}")

    all_true, all_prob, all_vid, all_subj = [], [], [], []
    for fi, (test_subj, train_all, test_ids) in enumerate(folds, 1):
        tr_subjects = sorted({by_vid.loc[v, "subject_id"] for v in train_all})
        rng = random.Random(cfg.get("seed", 42) + fi)
        rng.shuffle(tr_subjects)
        k = max(1, int(0.12 * len(tr_subjects)))
        val_s, tr_s = set(tr_subjects[:k]), set(tr_subjects[k:])
        train_ids = [v for v in train_all if by_vid.loc[v, "subject_id"] in tr_s]
        val_ids = [v for v in train_all if by_vid.loc[v, "subject_id"] in val_s]

        stats = _feature_stats(train_ids, fdir, adir)
        tr_anx = df[df.video_id.isin(train_ids)].anxiety
        a_mean, a_std = float(tr_anx.mean()), float(tr_anx.std())
        mk = lambda ids: MU3DDataset(meta, ids, fdir, adir, a_mean, a_std, stats, subj_to_idx)
        bs, nw = cfg["runtime"]["batch_size"], cfg["runtime"]["num_workers"]
        tl = DataLoader(mk(train_ids), batch_size=bs, shuffle=True, num_workers=nw, drop_last=True)
        vl = DataLoader(mk(val_ids), batch_size=bs, shuffle=False, num_workers=nw)
        test_ds = mk(test_ids)
        test_l = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=nw)

        model = MU3DDeceptionModel(cfg, n_subjects=len(subjects), da_cfg=da)
        loss_fn = MultiTaskLoss({"lie": cfg["multitask"]["lie_weight"],
                                 "anxiety": cfg["multitask"]["anxiety_weight"]})
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["optim"]["lr"],
                                weight_decay=cfg["optim"]["weight_decay"])
        trainer = MultiTaskTrainer(model, loss_fn, opt, device, cfg, ckpt_dir=cfg["paths"]["outputs"])
        trainer.train(tl, vl, epochs=cfg["optim"]["epochs"],
                      grad_accum=cfg["runtime"]["grad_accum_steps"],
                      patience=cfg["optim"]["early_stopping_patience"],
                      grl_lambda=grl_lambda, verbose=False)
        model.load_state_dict(torch.load(Path(cfg["paths"]["outputs"]) / "best.pt", map_location=device))

        vlog, vy = _logits(model, vl, device)
        ts = TemperatureScaler().to(device); ts.fit(vlog, vy)
        tlog, ty = _logits(model, test_l, device)
        prob = torch.sigmoid(ts(tlog)).detach().cpu().numpy()

        ordered_vids = test_ds.df.video_id.tolist()
        all_true.extend(ty.cpu().numpy().tolist())
        all_prob.extend(prob.tolist())
        all_vid.extend(ordered_vids)
        all_subj.extend([test_subj] * len(ordered_vids))
        print(f"[fold {fi:2d}/{len(folds)}] test_subj={test_subj}  "
              f"T={ts.temperature().item():.2f}  running_n={len(all_true)}")

    out = Path(cfg["paths"]["outputs"]); out.mkdir(parents=True, exist_ok=True)
    suffix = f"_{out_tag}" if out_tag else ""
    pred_path = out / f"loso_predictions{suffix}.csv"
    metrics_path = out / f"loso_metrics{suffix}.json"
    pd.DataFrame({"video_id": all_vid, "subject_id": all_subj,
                  "y_true": all_true, "y_prob": all_prob}).to_csv(pred_path, index=False)

    metrics = compute_metrics(all_true, all_prob)
    json.dump(metrics, open(metrics_path, "w"), indent=2)

    print(f"\nsaved predictions -> {pred_path}")
    print(f"=== LOSO results ({len(folds)} folds, {len(all_true)} clips){tag_note} ===")
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "eer", "ece"):
        print(f"  {key:10s} {metrics[key]:.4f}")
    return metrics