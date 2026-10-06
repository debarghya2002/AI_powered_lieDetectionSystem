"""Stage 5.4: total multi-task loss.

  L = w_lie * BCE(lie, veracity)
    + w_anx * SmoothL1(anxiety, anxiety_norm)
    [ + w_val * BCE(valence, valence) ]          # optional
    [ + w_att * SmoothL1(attractive, ...) ]      # optional
    [ + w_tru * SmoothL1(trustworthy, ...) ]     # optional
"""
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskLoss(nn.Module):
    def __init__(self, weights: dict):
        super().__init__()
        self.w = weights  # keys: lie, anxiety, (valence, attractive, trustworthy)

    def forward(self, outputs: dict, targets: dict):
        loss = self.w["lie"] * F.binary_cross_entropy_with_logits(
            outputs["lie"], targets["lie"].float())
        loss = loss + self.w["anxiety"] * F.smooth_l1_loss(
            outputs["anxiety"], targets["anxiety"].float())
        # if self.w.get("valence", 0) > 0:
        #     loss = loss + self.w["valence"] * F.binary_cross_entropy_with_logits(
        #         outputs["valence"], targets["valence"].float())
        # if self.w.get("attractive", 0) > 0:
        #     loss = loss + self.w["attractive"] * F.smooth_l1_loss(
        #         outputs["attractive"], targets["attractive"].float())
        # if self.w.get("trustworthy", 0) > 0:
        #     loss = loss + self.w["trustworthy"] * F.smooth_l1_loss(
        #         outputs["trustworthy"], targets["trustworthy"].float())
        return loss
