"""Load and merge YAML configs. default.yaml may list sibling files under
`defaults:` (e.g. preprocess, model, train) which are merged in."""
from pathlib import Path
import yaml


def load_config(path: str = "configs/default.yaml") -> dict:
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    for name in cfg.pop("defaults", []):
        sib = path.parent / f"{name}.yaml"
        if sib.exists():
            with open(sib) as f:
                part = yaml.safe_load(f) or {}
            for k, v in part.items():
                if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                    cfg[k].update(v)
                else:
                    cfg[k] = v
    return cfg
