"""Stage 6.3: adversarial domain classifier behind the GRL.

Predicts subject identity (and optionally sex/race) from the fused
representation. Because it sits behind a gradient-reversal layer, the backbone
is pushed to make these UN-predictable -> subject/demographic-invariant features,
so the model can't just memorize specific people. These are NOT prediction
heads; their job is to be defeated.
"""
import torch.nn as nn

from .grl import grad_reverse


def _head(in_dim, out_dim, hidden):
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim))


class DomainClassifier(nn.Module):
    def __init__(self, in_dim, n_subjects, use_subject: bool = True,
                 use_sex: bool = False, use_race: bool = False, hidden: int = 128):
        super().__init__()
        self.use_subject, self.use_sex, self.use_race = use_subject, use_sex, use_race
        if use_subject:
            self.subject = _head(in_dim, n_subjects, hidden)
        if use_sex:
            self.sex = _head(in_dim, 2, hidden)
        if use_race:
            self.race = _head(in_dim, 2, hidden)

    def forward(self, fused, grl_lambda: float = 1.0):
        x = grad_reverse(fused, grl_lambda)
        out = {}
        if self.use_subject:
            out["subject"] = self.subject(x)
        if self.use_sex:
            out["sex"] = self.sex(x)
        if self.use_race:
            out["race"] = self.race(x)
        return out