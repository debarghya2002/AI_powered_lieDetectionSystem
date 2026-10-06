"""Plot LOSO results from outputs/loso_predictions.csv.

Produces a 6-panel dashboard (ROC, Precision-Recall, calibration/reliability,
confusion matrix, score distribution, summary metrics) plus an individual PNG
for each panel and a per-subject accuracy plot, saved to outputs/figures/.

Run after LOSO:  python plot_results.py
Needs: matplotlib (pip install matplotlib).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                             average_precision_score, confusion_matrix)

from evaluation.metrics import compute_metrics, expected_calibration_error

PRED = Path("outputs/loso_predictions.csv")
FIGDIR = Path("outputs/figures")


def _roc(ax, y, p):
    fpr, tpr, _ = roc_curve(y, p)
    ax.plot(fpr, tpr, lw=2, color="#264653", label=f"AUC = {auc(fpr, tpr):.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve"); ax.legend(loc="lower right")


def _pr(ax, y, p):
    prec, rec, _ = precision_recall_curve(y, p)
    ax.plot(rec, prec, lw=2, color="#2a9d8f", label=f"AP = {average_precision_score(y, p):.3f}")
    ax.axhline(np.mean(y), ls="--", color="gray", lw=1, label="baseline")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall"); ax.legend(loc="lower left")


def _reliability(ax, y, p, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1); mids = (bins[:-1] + bins[1:]) / 2
    accs = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p > lo) & (p <= hi)
        accs.append((y[m] == (p[m] > 0.5)).mean() if m.sum() else np.nan)
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="perfect")
    ax.plot(mids, accs, "o-", lw=2, color="#e76f51",
            label=f"ECE = {expected_calibration_error(y, p, n_bins):.3f}")
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed accuracy")
    ax.set_title("Calibration (reliability)"); ax.legend(loc="upper left")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)


def _confusion(ax, y, p):
    cm = confusion_matrix(y, (p > 0.5).astype(int))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["lie", "truth"]); ax.set_yticklabels(["lie", "truth"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=13,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")


def _scores(ax, y, p):
    ax.hist(p[y == 1], bins=20, alpha=0.6, label="truth", color="#2a9d8f")
    ax.hist(p[y == 0], bins=20, alpha=0.6, label="lie", color="#e76f51")
    ax.axvline(0.5, ls="--", color="gray", lw=1)
    ax.set_xlabel("Predicted P(truth)"); ax.set_ylabel("Count")
    ax.set_title("Score distribution by class"); ax.legend()


def _metric_bars(ax, y, p):
    m = compute_metrics(y, p)
    keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    vals = [m[k] for k in keys]
    bars = ax.bar(keys, vals, color="#264653")
    ax.axhline(0.5, ls="--", color="gray", lw=1)
    ax.set_ylim(0, 1); ax.set_title("Summary metrics")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")


def _per_subject(ax, df):
    g = df.assign(correct=(df.y_prob > 0.5).astype(int) == df.y_true)
    acc = g.groupby("subject_id").correct.mean()
    ax.hist(acc, bins=np.linspace(0, 1, 6), color="#e9c46a", edgecolor="white")
    ax.axvline(acc.mean(), ls="--", color="#264653", lw=2, label=f"mean = {acc.mean():.2f}")
    ax.set_xlabel("Per-subject accuracy"); ax.set_ylabel("# subjects")
    ax.set_title("Accuracy across held-out subjects"); ax.legend()


def main():
    df = pd.read_csv(PRED)
    y = df.y_true.to_numpy().astype(int)
    p = df.y_prob.to_numpy().astype(float)
    FIGDIR.mkdir(parents=True, exist_ok=True)

    panels = [("roc", _roc), ("pr", _pr), ("calibration", _reliability),
              ("confusion", _confusion), ("scores", _scores), ("metrics", _metric_bars)]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for (_, fn), ax in zip(panels, axes.ravel()):
        fn(ax, y, p)
    fig.suptitle(f"MU3D LOSO results  (n = {len(y)} clips)", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIGDIR / "loso_dashboard.png", dpi=150); plt.close(fig)

    for name, fn in panels:
        f, a = plt.subplots(figsize=(5.5, 4.5)); fn(a, y, p); f.tight_layout()
        f.savefig(FIGDIR / f"loso_{name}.png", dpi=150); plt.close(f)

    if "subject_id" in df.columns:
        f, a = plt.subplots(figsize=(5.5, 4.5)); _per_subject(a, df); f.tight_layout()
        f.savefig(FIGDIR / "loso_per_subject.png", dpi=150); plt.close(f)

    print(f"saved figures to {FIGDIR}/  (loso_dashboard.png + individual panels)")


if __name__ == "__main__":
    main()