"""Stage 8.2: temperature scaling for probability calibration.

Learns a single scalar T (clamped to [0.5, 5]) that divides the logits so
predicted probabilities match observed accuracy. Fit on held-out logits by
minimizing NLL; applied to test logits before sigmoid. The clamp prevents the
calibrator from collapsing predictions toward 0.5 when the validation set is
tiny/noisy.
"""
import torch
import torch.nn as nn

T_MIN, T_MAX = 0.5, 5.0


class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_t = nn.Parameter(torch.zeros(1))   # T = exp(log_t)

    def temperature(self):
        return self.log_t.exp().clamp(T_MIN, T_MAX)

    def forward(self, logits):
        return logits / self.temperature()

    def fit(self, logits, labels, max_iter: int = 200):
        """logits, labels: 1-D tensors (binary). Optimizes T via LBFGS on BCE."""
        logits = logits.detach()
        labels = labels.detach().float()
        opt = torch.optim.LBFGS([self.log_t], lr=0.05, max_iter=max_iter)
        bce = nn.BCEWithLogitsLoss()

        def closure():
            opt.zero_grad()
            loss = bce(self.forward(logits), labels)
            loss.backward()
            return loss

        opt.step(closure)
        return self