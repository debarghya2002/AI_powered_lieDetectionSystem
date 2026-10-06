"""Stage 7: temporal cross-modal transformer.

Projects each modality's per-time-step tokens to d_model, concatenates them with
a CLS token into one sequence, adds modality-type + within-modality temporal
position embeddings, runs pre-norm self-attention over all tokens, and returns
the CLS output as the fused clip representation. Attention can now relate cues
ACROSS time and ACROSS modalities (e.g. a vocal-tension segment lining up with a
facial moment) -- exactly what the pooled model couldn't do.

Token layout: [CLS] + video(8) + landmark(16) + audio(16) + prosody(16) = 57.
"""
import torch
import torch.nn as nn

CLS, VIDEO, LANDMARK, AUDIO, PROSODY = 0, 1, 2, 3, 4
N_TYPES = 5


class TemporalCrossModalTransformer(nn.Module):
    def __init__(self, d_model: int = 256, n_heads: int = 4, n_layers: int = 2,
                 dropout: float = 0.3, video_dim: int = 768, landmark_dim: int = 1404,
                 audio_dim: int = 768, prosody_dim: int = 7, max_temporal: int = 16):
        super().__init__()
        self.proj_video = nn.Linear(video_dim, d_model)
        self.proj_landmark = nn.Linear(landmark_dim, d_model)
        self.proj_audio = nn.Linear(audio_dim, d_model)
        self.proj_prosody = nn.Linear(prosody_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        self.type_emb = nn.Embedding(N_TYPES, d_model)
        self.temporal_emb = nn.Embedding(max_temporal + 1, d_model)   # slot 0 = CLS
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.d_model = d_model

    @staticmethod
    def _ids(nv, nl, na, npr, device):
        type_ids = [CLS] + [VIDEO]*nv + [LANDMARK]*nl + [AUDIO]*na + [PROSODY]*npr
        pos = lambda n: list(range(1, n + 1))
        temp_ids = [0] + pos(nv) + pos(nl) + pos(na) + pos(npr)
        return (torch.tensor(type_ids, device=device),
                torch.tensor(temp_ids, device=device))

    def forward(self, video, landmark, audio, prosody):
        """video (B,8,768) landmark (B,16,1404) audio (B,16,768) prosody (B,16,7)."""
        B = video.size(0)
        v = self.proj_video(video)         # (B, 8, d)
        l = self.proj_landmark(landmark)   # (B, 16, d)
        a = self.proj_audio(audio)         # (B, 16, d)
        p = self.proj_prosody(prosody)     # (B, 16, d)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, v, l, a, p], dim=1)             # (B, 57, d)

        type_ids, temp_ids = self._ids(v.size(1), l.size(1), a.size(1), p.size(1), tokens.device)
        tokens = tokens + self.type_emb(type_ids) + self.temporal_emb(temp_ids)  # broadcast over B

        out = self.encoder(tokens)
        return self.norm(out[:, 0])        # CLS -> (B, d)