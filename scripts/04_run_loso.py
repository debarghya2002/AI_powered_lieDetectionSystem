"""Step 4: full LOSO train + evaluation across all 80 subjects.

  python scripts/04_run_loso.py            # all 80 folds (slow)
  python scripts/04_run_loso.py --folds 5  # quick smoke test on 5 folds
"""
import sys, pathlib, argparse
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from evaluation.loso_evaluator import run_loso

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--folds", type=int, default=None, help="run only the first N folds")
    a = p.parse_args()
    run_loso(a.config, n_folds=a.folds)