"""Print a full readout of the current canonical LOSO result:
outputs/loso_metrics.json + outputs/loso_predictions.csv.

Run:  python show_results.py
"""
import json
import pandas as pd

m = json.load(open("outputs/loso_metrics.json"))
print("=== metrics (outputs/loso_metrics.json) ===")
for k, v in m.items():
    print(f"  {k:10s} {v:.4f}")

df = pd.read_csv("outputs/loso_predictions.csv")
y, p = df.y_true.values, df.y_prob.values
pred = (p > 0.5).astype(int)
tp = int(((pred == 1) & (y == 1)).sum())
tn = int(((pred == 0) & (y == 0)).sum())
fp = int(((pred == 1) & (y == 0)).sum())
fn = int(((pred == 0) & (y == 1)).sum())

print(f"\n=== confusion matrix (n={len(df)} clips) ===")
print(f"  {'':16s}{'actual lie':>12s}{'actual truth':>14s}")
print(f"  {'predicted lie':16s}{tn:>12d}{fn:>14d}")
print(f"  {'predicted truth':16s}{fp:>12d}{tp:>14d}")

print(f"\n=== class-wise accuracy ===")
print(f"  lie clips   (y=0): n={int((y==0).sum())}  acc={(pred[y==0]==0).mean():.4f}")
print(f"  truth clips (y=1): n={int((y==1).sum())}  acc={(pred[y==1]==1).mean():.4f}")

sub = df.assign(correct=(pred == y)).groupby("subject_id").correct.mean()
print(f"\n=== per-subject accuracy (n={sub.shape[0]} subjects) ===")
print(f"  mean={sub.mean():.4f}  min={sub.min():.4f}  max={sub.max():.4f}")
print(f"  subjects at 100% correct: {int((sub==1.0).sum())}")
print(f"  subjects at 0%   correct: {int((sub==0.0).sum())}")
