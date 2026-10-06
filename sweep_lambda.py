"""Sweep grl_lambda (domain-adaptation strength) on a FAST reduced-fold LOSO to
screen candidate values before committing to a full 80-fold run.

Uses the SAME first N_FOLDS subjects for every lambda (fair, consistent
comparison) -- much cheaper than running 5 full 80-fold experiments. But it is
still a SCREENING test, not a final answer: exactly like the earlier 5-fold
smoke test that read 0.74 AUC and became 0.594 on the full run, reduced-fold
numbers are noisy. Whatever looks best here still needs a full 80-fold
confirmation run before you trust it or change your saved results.

SAFE BY DESIGN: this never touches outputs/loso_predictions.csv or your
_DA_on / _DA_off backups. Every lambda writes to its own tagged file
(outputs/loso_predictions_sweep_<lambda>.csv), so your current best
(lambda=0.5, acc=0.575, auc=0.594) stays exactly as-is no matter what this
script finds. Nothing to "revert" -- your saved results were never at risk.

Run:  python sweep_lambda.py
Tune N_FOLDS / LAMBDAS below for faster (fewer) or more thorough (more) runs.
"""
from evaluation.loso_evaluator import run_loso

N_FOLDS = 20                             # same first-20 subjects for every lambda
LAMBDAS = [0.1, 0.3, 0.5, 0.7, 1.0]       # 0.5 = your current full-LOSO best


def main():
    results = {}
    for lam in LAMBDAS:
        print(f"\n{'='*50}\nlambda = {lam}\n{'='*50}")
        m = run_loso("configs/default.yaml", n_folds=N_FOLDS,
                     grl_lambda_override=lam, out_tag=f"sweep_{lam}")
        results[lam] = m

    print(f"\n{'lambda':>8s} {'acc':>7s} {'auc':>7s} {'f1':>7s} {'ece':>7s}")
    print("-" * 40)
    for lam, m in results.items():
        print(f"{lam:8.2f} {m['accuracy']:7.3f} {m['roc_auc']:7.3f} {m['f1']:7.3f} {m['ece']:7.3f}")

    best = max(results, key=lambda k: results[k]["roc_auc"])
    print(f"\nbest by AUC on this {N_FOLDS}-fold screen: lambda={best} "
          f"(AUC={results[best]['roc_auc']:.3f})")
    print("This is a REDUCED-FOLD screen -- confirm the winner with a full")
    print("80-fold LOSO run before adopting it over your current best:")
    print(f'  python -c "from evaluation.loso_evaluator import run_loso; '
          f'run_loso(\'configs/default.yaml\', grl_lambda_override={best}, out_tag=\'confirm_{best}\')"')


if __name__ == "__main__":
    main()