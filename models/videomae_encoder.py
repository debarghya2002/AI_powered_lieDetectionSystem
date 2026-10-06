"""Stage 2A: pretrained VideoMAE -> TEMPORAL sequence of visual tokens.

FROZEN feature extractor (no training). Given 16 uniformly sampled RGB frames of
one clip, returns an (8, 768) sequence: VideoMAE yields 8 temporal positions
(16 frames / tubelet 2), and we mean-pool over the 14x14 spatial patches at each
temporal position to keep the time axis. Cached to disk; the trainable temporal
model consumes the sequence later.

VideoMAE token order is temporal-major (all spatial patches of t=0, then t=1,
...), so reshaping (tokens, 768) -> (n_temporal, spatial, 768) and pooling the
spatial axis recovers the per-time-step embeddings.
"""
from typing import List
import numpy as np
import torch
import torch.nn as nn
from transformers import VideoMAEImageProcessor, VideoMAEModel


class VideoMAEEncoder(nn.Module):
    def __init__(self, pretrained: str = "MCG-NJU/videomae-base", device: str = "cpu"):
        super().__init__()
        self.processor = VideoMAEImageProcessor.from_pretrained(pretrained)
        self.model = VideoMAEModel.from_pretrained(pretrained).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.device = device
        c = self.model.config
        self.spatial = (c.image_size // c.patch_size) ** 2     # 196 for 224/16

    @torch.no_grad()
    def forward(self, frames: List[np.ndarray]) -> np.ndarray:
        """frames: list of 16 RGB uint8 arrays (H,W,3). Returns (n_temporal, 768) float32."""
        inputs = self.processor(list(frames), return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)   # (1, 16, C, H, W)
        hidden = self.model(pixel_values).last_hidden_state      # (1, tokens, 768)
        hidden = hidden.squeeze(0)                               # (tokens, 768)
        n_temporal = hidden.shape[0] // self.spatial            # 1568 / 196 = 8
        seq = hidden.view(n_temporal, self.spatial, hidden.shape[-1]).mean(dim=1)
        return seq.cpu().numpy().astype("float32")              # (n_temporal, 768)