"""Stage 3.2-3.3: pretrained wav2vec2 -> TEMPORAL sequence of speech tokens.

FROZEN feature extractor (no training). Given a 16 kHz mono waveform, returns an
(n_segments, 768) sequence: frame-level wav2vec2 outputs (~50/s) are concatenated
across chunks, then average-pooled into n_segments equal time bins to keep the
time axis at a fixed length. Cached; the trainable temporal model consumes it.

First use downloads the weights (~360 MB) from Hugging Face, then cached locally.
"""
import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model


class Wav2Vec2Encoder(nn.Module):
    def __init__(self, pretrained: str = "facebook/wav2vec2-base-960h",
                 device: str = "cpu", sr: int = 16000, chunk_seconds: int = 20,
                 n_segments: int = 16):
        super().__init__()
        self.fe = Wav2Vec2FeatureExtractor.from_pretrained(pretrained)
        self.model = Wav2Vec2Model.from_pretrained(pretrained).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.device = device
        self.sr = sr
        self.chunk = int(chunk_seconds * sr)
        self.n_segments = n_segments
        self.dim = self.model.config.hidden_size   # 768

    @staticmethod
    def _segment_pool(x: torch.Tensor, n: int) -> torch.Tensor:
        """x: (T, D) -> (n, D) by averaging T frames into n contiguous time bins."""
        T = x.shape[0]
        if T < n:                                   # pad short clips with last frame
            x = torch.cat([x, x[-1:].repeat(n - T, 1)], dim=0)
            T = n
        edges = torch.linspace(0, T, n + 1).long()
        segs = []
        for i in range(n):
            lo, hi = int(edges[i]), int(edges[i + 1])
            if hi <= lo:
                hi = lo + 1
            segs.append(x[lo:hi].mean(dim=0))
        return torch.stack(segs)

    @torch.no_grad()
    def forward(self, wav: np.ndarray) -> np.ndarray:
        """wav: 1-D float32 @ 16 kHz. Returns (n_segments, 768) float32."""
        if wav is None or len(wav) == 0:
            return np.zeros((self.n_segments, self.dim), dtype="float32")
        frames = []
        for start in range(0, len(wav), self.chunk):
            chunk = wav[start:start + self.chunk]
            if len(chunk) < self.sr // 10:          # skip <0.1s tails
                continue
            inp = self.fe(chunk, sampling_rate=self.sr,
                          return_tensors="pt").input_values.to(self.device)
            h = self.model(inp).last_hidden_state.squeeze(0)   # (T_frames, 768)
            frames.append(h)
        if not frames:
            return np.zeros((self.n_segments, self.dim), dtype="float32")
        allframes = torch.cat(frames, dim=0)                   # (T_total, 768)
        return self._segment_pool(allframes, self.n_segments).cpu().numpy().astype("float32")