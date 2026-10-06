"""Stage 3.4: prosodic features.

extract_prosody_seq -> (n_segments, 7) prosodic CONTOUR: per time-segment
                       [f0_mean, f0_std, voiced_frac, rms_mean, rms_std,
                        zcr_mean, centroid_mean]. librosa only.
"""
import numpy as np
import librosa

PROSODY_SEQ_DIM = 7       # per-segment contour
HOP = 512


def extract_prosody_seq(wav: np.ndarray, sr: int = 16000, n_segments: int = 16) -> np.ndarray:
    """Per-time-segment prosodic contour -> (n_segments, PROSODY_SEQ_DIM)."""
    k = PROSODY_SEQ_DIM
    if wav is None or len(wav) < sr // 5:
        return np.zeros((n_segments, k), dtype="float32")
    f0 = librosa.yin(wav, fmin=65, fmax=400, sr=sr, hop_length=HOP)
    rms = librosa.feature.rms(y=wav, hop_length=HOP)[0]
    zcr = librosa.feature.zero_crossing_rate(wav, hop_length=HOP)[0]
    cen = librosa.feature.spectral_centroid(y=wav, sr=sr, hop_length=HOP)[0]
    n = min(len(f0), len(rms), len(zcr), len(cen))
    f0, rms, zcr, cen = f0[:n], rms[:n], zcr[:n], cen[:n]
    voiced = rms > (0.5 * np.median(rms) + 1e-8)

    edges = np.linspace(0, n, n_segments + 1).astype(int)
    out = np.zeros((n_segments, k), dtype="float32")
    for i in range(n_segments):
        lo, hi = edges[i], max(edges[i] + 1, edges[i + 1])
        sl = slice(lo, min(hi, n))
        f0s, rs, vs, zs, cs = f0[sl], rms[sl], voiced[sl], zcr[sl], cen[sl]
        f0v = f0s[vs & np.isfinite(f0s)] if f0s.size else f0s
        out[i] = [
            float(np.mean(f0v)) if f0v.size else 0.0,
            float(np.std(f0v)) if f0v.size else 0.0,
            float(np.mean(vs)) if vs.size else 0.0,
            float(np.mean(rs)) if rs.size else 0.0,
            float(np.std(rs)) if rs.size else 0.0,
            float(np.mean(zs)) if zs.size else 0.0,
            float(np.mean(cs)) if cs.size else 0.0,
        ]
    return out