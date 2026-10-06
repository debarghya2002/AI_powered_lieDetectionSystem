"""Stage 5/6/7: full trainable TEMPORAL model.

Frozen backbones (VideoMAE, wav2vec2) are cached as sequences; this consumes the
four per-clip sequences, fuses them with the temporal cross-modal transformer,
and returns (head outputs, fused CLS rep). The optional domain classifier
(Stage 6) is held as self.domain and called by the trainer on the fused rep.
"""
import torch.nn as nn

from models.prosodic_features import PROSODY_SEQ_DIM
from fusion.temporal_cross_modal import TemporalCrossModalTransformer
from multitask.heads import MultiTaskHeads
from domain_adaptation.domain_classifier import DomainClassifier

VIDEO_DIM = 768
AUDIO_DIM = 768
LANDMARK_DIM = 468 * 3      # 1404


class MU3DDeceptionModel(nn.Module):
    def __init__(self, cfg: dict, n_subjects: int = 0, da_cfg: dict | None = None):
        super().__init__()
        cmt = cfg["cross_modal_transformer"]
        self.transformer = TemporalCrossModalTransformer(
            d_model=cmt["d_model"], n_heads=cmt["n_heads"], n_layers=cmt["n_layers"],
            dropout=cmt["dropout"], video_dim=VIDEO_DIM, landmark_dim=LANDMARK_DIM,
            audio_dim=AUDIO_DIM, prosody_dim=PROSODY_SEQ_DIM, max_temporal=16)
        self.heads = MultiTaskHeads(cmt["d_model"])

        self.domain = None
        if da_cfg and da_cfg.get("enabled") and n_subjects > 0:
            self.domain = DomainClassifier(
                cmt["d_model"], n_subjects,
                da_cfg.get("use_subject", True),
                da_cfg.get("use_sex", False),
                da_cfg.get("use_race", False))

    def forward(self, videomae, landmarks, wav2vec, prosody):
        fused = self.transformer(videomae, landmarks, wav2vec, prosody)
        return self.heads(fused), fused