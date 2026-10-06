"""Stage 2B.1-2: MediaPipe FaceMesh landmarks per frame + normalization.

DETERMINISTIC. Produces a (T, 468*3) sequence per clip, cached and later fed to
the trainable LandmarkBiLSTM. Normalization removes face position (center on
centroid) and scale (divide by interocular distance) so the model sees facial
motion, not where the face sits in the frame. Frames with no detected face stay
zero.

Uses the standard MediaPipe `solutions.face_mesh` API (bundled model, no
download). One FaceMesh instance is reused across all clips for speed.
"""
import numpy as np
import mediapipe as mp

NUM_LANDMARKS = 468
LEFT_EYE_CORNER, RIGHT_EYE_CORNER = 33, 263   # interocular scale reference

_FACE_MESH = None


def _get_face_mesh():
    """Lazily build and cache a single FaceMesh instance."""
    global _FACE_MESH
    if _FACE_MESH is None:
        if not hasattr(mp, "solutions"):
            raise RuntimeError(
                "This MediaPipe build lacks the legacy `solutions` API "
                "(face_mesh). Install the standard wheel: `pip install mediapipe`."
            )
        _FACE_MESH = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, refine_landmarks=False)
    return _FACE_MESH


def _normalize(pts: np.ndarray) -> np.ndarray:
    """pts: (468, 3) -> centered on centroid, scaled by interocular distance."""
    pts = pts - pts.mean(axis=0, keepdims=True)
    scale = np.linalg.norm(pts[LEFT_EYE_CORNER] - pts[RIGHT_EYE_CORNER]) + 1e-6
    return pts / scale


def extract_landmarks(frames, num_landmarks: int = NUM_LANDMARKS) -> np.ndarray:
    """frames: list of N RGB uint8 arrays. Returns (N, num_landmarks*3) float32."""
    fm = _get_face_mesh()
    seq = np.zeros((len(frames), num_landmarks * 3), dtype="float32")
    for i, frame in enumerate(frames):
        res = fm.process(frame)
        if not res.multi_face_landmarks:
            continue
        lm = res.multi_face_landmarks[0].landmark
        pts = np.array([[p.x, p.y, p.z] for p in lm], dtype="float32")  # (468,3)
        seq[i] = _normalize(pts).reshape(-1)
    return seq