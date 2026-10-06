"""Prediction-only runner for one saved video file.

This module has no webcam, microphone, or live-streaming code.  It loads the
already-trained model from an inference bundle, extracts features from the
chosen video file, and returns one prediction without changing any weights.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import librosa
import numpy as np
import torch

from models.deception_model import MU3DDeceptionModel
from models.landmark_extractor import extract_landmarks
from models.prosodic_features import extract_prosody_seq
from models.videomae_encoder import VideoMAEEncoder
from models.wav2vec_encoder import Wav2Vec2Encoder
from preprocessing.audio_extractor import extract_audio
from utils.video_frames import sample_frames


def _torch_load(path: Path, device: torch.device):
    """Load a tensor-only bundle on both new and older PyTorch versions."""
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # PyTorch < 2.0
        return torch.load(path, map_location=device)


@dataclass(frozen=True)
class VideoPrediction:
    """Prediction for one complete saved video file."""

    status: str
    deception_probability: float | None = None
    truth_probability: float | None = None
    face_fraction: float | None = None
    rms: float | None = None


class VideoFilePredictor:
    """Run the frozen feature extractors and saved MU3D classifier on one file."""

    def __init__(self, bundle_path: str | Path, device: str = "auto",
                 min_speech_rms: float = 0.003, min_face_fraction: float = 0.50):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.bundle_path = Path(bundle_path)
        if not self.bundle_path.exists():
            raise FileNotFoundError(
                f"Prediction bundle not found: {self.bundle_path}. "
                "Run scripts/create_inference_bundle.py first.")
        self.bundle = _torch_load(self.bundle_path, self.device)
        required = {"model_state", "model_config", "feature_stats", "temperature",
                    "extractors", "n_subjects", "domain_adaptation"}
        missing = required - set(self.bundle)
        if missing:
            raise ValueError(f"Invalid inference bundle; missing {sorted(missing)}")

        self.model = MU3DDeceptionModel(
            self.bundle["model_config"],
            n_subjects=int(self.bundle["n_subjects"]),
            da_cfg=self.bundle["domain_adaptation"],
        ).to(self.device)
        incompatible = self.model.load_state_dict(self.bundle["model_state"], strict=False)
        wrong_missing = [key for key in incompatible.missing_keys if not key.startswith("domain.")]
        wrong_extra = [key for key in incompatible.unexpected_keys if not key.startswith("domain.")]
        if wrong_missing or wrong_extra:
            raise ValueError(
                f"Bundle/model mismatch; missing={wrong_missing}, unexpected={wrong_extra}")
        self.model.eval()

        extractor_cfg = self.bundle["extractors"]
        self.sample_rate = int(extractor_cfg["sample_rate"])
        self.num_video_frames = int(extractor_cfg["num_video_frames"])
        self.num_audio_segments = int(extractor_cfg["num_audio_segments"])
        self.video_encoder = VideoMAEEncoder(extractor_cfg["videomae_pretrained"],
                                             device=str(self.device))
        self.audio_encoder = Wav2Vec2Encoder(
            extractor_cfg["wav2vec_pretrained"], device=str(self.device),
            sr=self.sample_rate, n_segments=self.num_audio_segments)
        self.stats = self.bundle["feature_stats"]
        self.temperature = max(float(self.bundle["temperature"]), 1e-6)
        self.min_speech_rms = float(min_speech_rms)
        self.min_face_fraction = float(min_face_fraction)

    def _standardize(self, name: str, values: np.ndarray) -> np.ndarray:
        stat = self.stats[name]

        def to_numpy(value):
            return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)

        mean = np.asarray(to_numpy(stat["mean"]), dtype="float32")
        std = np.asarray(to_numpy(stat["std"]), dtype="float32")
        return ((values.astype("float32") - mean) / np.maximum(std, 1e-6)).astype("float32")

    @torch.inference_mode()
    def _predict_features(self, frames_rgb, wav: np.ndarray) -> VideoPrediction:
        if len(frames_rgb) != self.num_video_frames:
            return VideoPrediction("could not sample enough video frames")
        if len(wav) < self.sample_rate // 2:
            return VideoPrediction("video has too little audio")

        wav = np.asarray(wav, dtype="float32").reshape(-1)
        rms = float(np.sqrt(np.mean(np.square(wav)) + 1e-12))
        if rms < self.min_speech_rms:
            return VideoPrediction("no clear speech in this video", rms=rms)

        landmarks = extract_landmarks(frames_rgb)
        face_fraction = float(np.mean(np.any(landmarks != 0.0, axis=1)))
        if face_fraction < self.min_face_fraction:
            return VideoPrediction("face not consistently visible", face_fraction=face_fraction, rms=rms)

        videomae = self._standardize("videomae", self.video_encoder(list(frames_rgb)))
        wav2vec = self._standardize("wav2vec", self.audio_encoder(wav))
        prosody = self._standardize(
            "prosody", extract_prosody_seq(wav, sr=self.sample_rate,
                                            n_segments=self.num_audio_segments))
        to_tensor = lambda values: torch.from_numpy(values).unsqueeze(0).to(self.device)
        output, _ = self.model(to_tensor(videomae), to_tensor(landmarks),
                               to_tensor(wav2vec), to_tensor(prosody))
        truth_probability = float(torch.sigmoid(output["lie"] / self.temperature).item())
        return VideoPrediction(
            status="ok",
            truth_probability=truth_probability,
            deception_probability=1.0 - truth_probability,
            face_fraction=face_fraction,
            rms=rms,
        )

    def predict(self, video_path: str | Path) -> VideoPrediction:
        """Extract features from an MP4/WMV/etc. video and return its prediction."""
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        frames = sample_frames(video_path, num_frames=self.num_video_frames)
        with TemporaryDirectory(prefix="mu3d_predict_") as temp_dir:
            wav_path = Path(temp_dir) / "audio.wav"
            extract_audio(video_path, wav_path, sample_rate=self.sample_rate)
            wav, _ = librosa.load(str(wav_path), sr=self.sample_rate, mono=True)
        return self._predict_features(frames, wav.astype("float32"))
