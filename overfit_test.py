"""Diagnostic: can the TEMPORAL model overfit a tiny balanced subset (dropout
off)? If train acc reaches ~1.0, the temporal architecture trains correctly and
any chance-level LOSO is a generalization issue (expected). If it cannot, the
temporal wiring has a problem.

Run:  python overfit_test.py     (or: python -c "import overfit_test; overfit_test.main()")
"""
import copy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.config import load_config
from utils.seed import set_seed
from utils.device import get_device
from datasets.mu3d_dataset import MU3DDataset
from models.deception_model import MU3DDeceptionModel


def feature_stats(ids, facial_dir, audio_dir):
    def stack(d, suffix):
        return np.stack([np.load(Path(d) / f"{v}_{suffix}.npy") for v in ids])
    ms = lambda a: (a.reshape(-1, a.shape[-1]).mean(0).astype("float32"),
                    (a.reshape(-1, a.shape[-1]).std(0) + 1e-6).astype("float32"))
    return {"videomae": ms(stack(facial_dir, "videomae_seq")),
            "wav2vec":  ms(stack(audio_dir, "wav2vec_seq")),
            "prosody":  ms(stack(audio_dir, "prosody_seq"))}


def main():
    cfg = load_config("configs/default.yaml")
    set_seed(42)
    device = get_device("auto")

    df = pd.read_csv(cfg["paths"]["metadata"])
    ids = (df[df.label == 0].video_id.tolist()[:16] +
           df[df.label == 1].video_id.tolist()[:16])      # 16 lies + 16 truths
    fdir, adir = cfg["paths"]["embeddings_facial"], cfg["paths"]["embeddings_audio"]
    stats = feature_stats(ids, fdir, adir)
    ds = MU3DDataset(cfg["paths"]["metadata"], ids, fdir, adir, 0.0, 1.0, stats)
    loader = DataLoader(ds, batch_size=8, shuffle=True)

    cfg2 = copy.deepcopy(cfg)                              # dropout OFF
    cfg2["cross_modal_transformer"]["dropout"] = 0.0
    model = MU3DDeceptionModel(cfg2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)

    print(f"overfitting {len(ids)} clips (dropout off) on {device}")
    for ep in range(1, 301):
        model.train()
        for b in loader:
            b = {k: v.to(device) for k, v in b.items()}
            out, _ = model(b["videomae"], b["landmarks"], b["wav2vec"], b["prosody"])
            loss = F.binary_cross_entropy_with_logits(out["lie"], b["label"])
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 25 == 0:
            model.eval()
            correct = n = 0
            tot = 0.0
            with torch.no_grad():
                for b in loader:
                    b = {k: v.to(device) for k, v in b.items()}
                    out, _ = model(b["videomae"], b["landmarks"], b["wav2vec"], b["prosody"])
                    tot += F.binary_cross_entropy_with_logits(out["lie"], b["label"]).item() * b["label"].size(0)
                    pred = (torch.sigmoid(out["lie"]) > 0.5).float()
                    correct += (pred == b["label"]).sum().item(); n += b["label"].size(0)
            print(f"epoch {ep:3d} | train_loss {tot/n:.4f} | train_acc {correct/n:.4f}")


if __name__ == "__main__":
    main()