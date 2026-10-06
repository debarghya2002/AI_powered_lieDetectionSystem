"""Stage 8.3: Monte-Carlo dropout for predictive uncertainty.

Keeps dropout active at inference (this model has no BatchNorm, so train() only
turns on dropout), runs several stochastic passes, and returns the mean lie
probability plus its std as an uncertainty estimate.
"""
import torch


@torch.no_grad()
def mc_dropout_predict(model, inputs, n_passes: int = 20):
    """inputs: (videomae, landmarks, wav2vec, prosody) tensors on the model's
    device. Returns (mean_prob, std) each shape (B,)."""
    model.train()
    probs = []
    for _ in range(n_passes):
        out, _ = model(*inputs)
        probs.append(torch.sigmoid(out["lie"]))
    probs = torch.stack(probs)              # (n_passes, B)
    return probs.mean(0), probs.std(0)