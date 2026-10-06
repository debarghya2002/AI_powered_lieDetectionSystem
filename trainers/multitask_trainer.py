"""Stage 5/6 trainer: multi-task (lie + anxiety) loop with optional
domain-adversarial loss (GRL), grad accumulation, early stopping on val
accuracy, best-checkpoint saving. verbose=False silences per-epoch logging
(used by the LOSO loop).
"""
import torch
import torch.nn.functional as F

from .base_trainer import BaseTrainer, EarlyStopping


class MultiTaskTrainer(BaseTrainer):
    def _move(self, batch):
        return {k: v.to(self.device) for k, v in batch.items()}

    @staticmethod
    def _targets(batch):
        return {"lie": batch["label"], "anxiety": batch["anxiety"]}

    def train(self, train_loader, val_loader, epochs=50, grad_accum=1,
              patience=8, grl_lambda=0.0, verbose=True):
        stopper = EarlyStopping(patience=patience, mode="max")
        history = []
        for ep in range(1, epochs + 1):
            self.model.train()
            self.opt.zero_grad()
            tr_loss, tr_n, tr_correct = 0.0, 0, 0
            for i, batch in enumerate(train_loader):
                batch = self._move(batch)
                out, fused = self.model(batch["videomae"], batch["landmarks"],
                                        batch["wav2vec"], batch["prosody"])
                loss = self.loss_fn(out, self._targets(batch))
                if getattr(self.model, "domain", None) is not None and grl_lambda > 0:
                    dom = self.model.domain(fused, grl_lambda)
                    loss = loss + sum(F.cross_entropy(v, batch[k].long())
                                      for k, v in dom.items())
                (loss / grad_accum).backward()
                if (i + 1) % grad_accum == 0:
                    self.opt.step()
                    self.opt.zero_grad()
                bs = batch["label"].size(0)
                tr_loss += loss.item() * bs
                tr_n += bs
                pred = (torch.sigmoid(out["lie"].detach()) > 0.5).float()
                tr_correct += (pred == batch["label"]).sum().item()
            self.opt.step()
            self.opt.zero_grad()

            tr_loss, tr_acc = tr_loss / max(tr_n, 1), tr_correct / max(tr_n, 1)
            val_loss, val_acc = self.evaluate(val_loader)
            improved = stopper.step(val_acc)
            if improved:
                self.save("best.pt")
            history.append((ep, tr_loss, tr_acc, val_loss, val_acc))
            if verbose:
                print(f"epoch {ep:02d} | train_loss {tr_loss:.4f} acc {tr_acc:.4f} | "
                      f"val_loss {val_loss:.4f} acc {val_acc:.4f}{'  *' if improved else ''}")
            if stopper.should_stop:
                if verbose:
                    print(f"early stop at epoch {ep} (best val_acc {stopper.best:.4f})")
                break
        return history, stopper.best

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        tot_loss, n, correct = 0.0, 0, 0
        for batch in loader:
            batch = self._move(batch)
            out, _ = self.model(batch["videomae"], batch["landmarks"],
                                batch["wav2vec"], batch["prosody"])
            loss = self.loss_fn(out, self._targets(batch))
            bs = batch["label"].size(0)
            tot_loss += loss.item() * bs
            n += bs
            pred = (torch.sigmoid(out["lie"]) > 0.5).float()
            correct += (pred == batch["label"]).sum().item()
        return tot_loss / max(n, 1), correct / max(n, 1)