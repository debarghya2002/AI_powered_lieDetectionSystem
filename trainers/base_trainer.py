"""Shared training utilities: checkpoint saving and early stopping."""
from pathlib import Path
import torch


class EarlyStopping:
    def __init__(self, patience: int = 8, mode: str = "max"):
        self.patience, self.mode = patience, mode
        self.best, self.count, self.should_stop = None, 0, False

    def step(self, metric: float) -> bool:
        better = self.best is None or (
            metric > self.best if self.mode == "max" else metric < self.best)
        if better:
            self.best, self.count = metric, 0
            return True
        self.count += 1
        self.should_stop = self.count >= self.patience
        return False


class BaseTrainer:
    def __init__(self, model, loss_fn, optimizer, device, cfg, ckpt_dir="outputs"):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.opt = optimizer
        self.device = device
        self.cfg = cfg
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def save(self, name: str = "best.pt"):
        torch.save(self.model.state_dict(), self.ckpt_dir / name)