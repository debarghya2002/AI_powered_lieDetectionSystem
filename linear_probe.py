"""Diagnostic: logistic-regression LINEAR PROBE on the frozen features, under
the same LOSO protocol. Answers whether the cached features carry deception
signal that transfers across subjects -- and which modality -- independent of the
deep model. Fast: sklearn LR, no training loops.

If every row is ~0.50 AUC, the features are the ceiling (focus on better
features / temporal modeling). If some row is clearly higher, the deep model is
the bottleneck, not the features.

Run:  python linear_probe.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from utils.config import load_config
from datasets.loso_splitter import generate_loso_folds
from evaluation.metrics import compute_metrics


def load_feature(vid, fdir, adir, which):
    f = lambda d, s: np.load(Path(d) / f"{vid}_{s}.npy")
    if which == "videomae":
        return f(fdir, "videomae")
    if which == "landmarks":
        return f(fdir, "landmarks").mean(0)          # pool over time -> (1404,)
    if which == "wav2vec":
        return f(adir, "wav2vec")
    if which == "prosody":
        return f(adir, "prosody")
    if which == "all":
        return np.concatenate([f(fdir, "videomae"), f(fdir, "landmarks").mean(0),
                               f(adir, "wav2vec"), f(adir, "prosody")])
    raise ValueError(which)


def probe(which, fdir, adir, folds, label_by_vid):
    y_true, y_prob = [], []
    for _test_subj, train_ids, test_ids in folds:
        Xtr = np.stack([load_feature(v, fdir, adir, which) for v in train_ids])
        ytr = np.array([label_by_vid[v] for v in train_ids])
        Xte = np.stack([load_feature(v, fdir, adir, which) for v in test_ids])
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, C=0.1))
        clf.fit(Xtr, ytr)
        y_prob.extend(clf.predict_proba(Xte)[:, 1].tolist())
        y_true.extend(label_by_vid[v] for v in test_ids)
    return compute_metrics(y_true, y_prob)


def main():
    cfg = load_config("configs/default.yaml")
    meta = cfg["paths"]["metadata"]
    fdir, adir = cfg["paths"]["embeddings_facial"], cfg["paths"]["embeddings_audio"]
    df = pd.read_csv(meta)
    label_by_vid = dict(zip(df.video_id, df.label))
    folds = list(generate_loso_folds(meta))

    print(f"linear probe | LOSO {len(folds)} folds, {len(df)} clips\n")
    print(f"{'feature set':14s} {'AUC':>7s} {'acc':>7s} {'f1':>7s}")
    print("-" * 38)
    for which in ["videomae", "landmarks", "wav2vec", "prosody", "all"]:
        m = probe(which, fdir, adir, folds, label_by_vid)
        print(f"{which:14s} {m['roc_auc']:7.3f} {m['accuracy']:7.3f} {m['f1']:7.3f}")


if __name__ == "__main__":
    main()