"""Compare LOSO results WITH vs WITHOUT domain adaptation.

Reads the two pooled-prediction CSVs, prints a side-by-side metric table, and
saves an overlaid ROC plot. Produce both runs first:
    outputs/loso_predictions_DA_on.csv   (domain adaptation ON)
    outputs/loso_predictions.csv         (domain adaptation OFF, latest run)

Run:  python compare_da.py
"""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from evaluation.metrics import compute_metrics

RUNS = {"DA on":  Path("outputs/loso_predictions_DA_on.csv"),
        "DA off": Path("outputs/loso_predictions.csv")}


def main():
    data = {}
    for name, path in RUNS.items():
        if not path.exists():
            print(f"MISSING: {path}\n-> run that configuration first (see instructions).")
            return
        df = pd.read_csv(path)
        data[name] = (df.y_true.to_numpy().astype(int), df.y_prob.to_numpy().astype(float))

    mon = compute_metrics(*data["DA on"])
    moff = compute_metrics(*data["DA off"])
    keys = ["accuracy", "precision", "recall", "f1", "roc_auc", "eer", "ece"]
    print(f"{'metric':10s} {'DA on':>8s} {'DA off':>8s} {'on - off':>9s}")
    print("-" * 39)
    for k in keys:
        print(f"{k:10s} {mon[k]:8.3f} {moff[k]:8.3f} {mon[k]-moff[k]:+9.3f}")
    print("\n(higher better for all except eer & ece; for those a NEGATIVE 'on-off' means DA helped)")

    plt.figure(figsize=(6, 5))
    for name, (y, p) in data.items():
        fpr, tpr, _ = roc_curve(y, p)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("LOSO ROC: domain adaptation on vs off")
    plt.legend(loc="lower right")
    out = Path("outputs/figures"); out.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(out / "da_comparison_roc.png", dpi=150); plt.close()
    print(f"saved overlaid ROC -> {out / 'da_comparison_roc.png'}")


if __name__ == "__main__":
    main()