"""Stage 5: multi-task prediction heads on the shared fused representation.

DEFAULT = two heads:
  - LieHead     : primary, binary (veracity), BCEWithLogits
  - AnxietyHead : auxiliary, regression on normalized 1-7 anxiety, Smooth-L1
                  (orthogonal to the label, r=-0.04 -> a clean regularizer)

OPTIONAL heads are written but commented out. To enable one:
  1. uncomment its class instantiation in MultiTaskHeads.__init__
  2. uncomment its forward output below
  3. set the matching weight > 0 in configs/train.yaml
  4. uncomment its loss term in multitask/multitask_loss.py
"""
import torch.nn as nn


class LieHead(nn.Module):
    """Primary lie-detection head (binary)."""
    def __init__(self, in_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x):
        return self.fc(x).squeeze(-1)          # logits, BCEWithLogitsLoss


class AnxietyHead(nn.Module):
    """Auxiliary anxiety regression head (normalized 1-7 -> scalar)."""
    def __init__(self, in_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x):
        return self.fc(x).squeeze(-1)          # scalar, SmoothL1Loss


# ----- OPTIONAL auxiliary heads (disabled by default) -----
# class ValenceHead(nn.Module):
#     """Optional binary head: statement valence (clean, fully crossed)."""
#     def __init__(self, in_dim: int):
#         super().__init__()
#         self.fc = nn.Linear(in_dim, 1)
#     def forward(self, x):
#         return self.fc(x).squeeze(-1)        # logits, BCEWithLogitsLoss
#
# class AttractiveHead(nn.Module):
#     """Optional regression head: perceived attractiveness (1-7)."""
#     def __init__(self, in_dim: int):
#         super().__init__()
#         self.fc = nn.Linear(in_dim, 1)
#     def forward(self, x):
#         return self.fc(x).squeeze(-1)        # scalar, SmoothL1Loss
#
# class TrustworthyHead(nn.Module):
#     """Optional regression head: perceived trustworthiness (1-7)."""
#     def __init__(self, in_dim: int):
#         super().__init__()
#         self.fc = nn.Linear(in_dim, 1)
#     def forward(self, x):
#         return self.fc(x).squeeze(-1)        # scalar, SmoothL1Loss


class MultiTaskHeads(nn.Module):
    """Container holding the active heads."""
    def __init__(self, in_dim: int):
        super().__init__()
        self.lie = LieHead(in_dim)
        self.anxiety = AnxietyHead(in_dim)
        # self.valence = ValenceHead(in_dim)
        # self.attractive = AttractiveHead(in_dim)
        # self.trustworthy = TrustworthyHead(in_dim)

    def forward(self, shared):
        out = {
            "lie": self.lie(shared),
            "anxiety": self.anxiety(shared),
            # "valence": self.valence(shared),
            # "attractive": self.attractive(shared),
            # "trustworthy": self.trustworthy(shared),
        }
        return out
