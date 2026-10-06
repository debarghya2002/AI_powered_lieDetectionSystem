"""Stage 1 driver: for each video in master_metadata.csv, normalize the video
and extract 16kHz mono audio. One item at a time (memory-safe), resumable
(skips files already produced)."""
from pathlib import Path
import pandas as pd
try:
    from tqdm import tqdm
except ImportError:  # tqdm optional
    def tqdm(it, **k):
        return it

from utils.config import load_config
from utils.logging_utils import get_logger
from preprocessing.video_normalizer import normalize_video
from preprocessing.audio_extractor import extract_audio

RAW_EXT = ".wmv"


def run(config_path: str = "configs/default.yaml") -> None:
    cfg = load_config(config_path)
    log = get_logger("preprocess", cfg["paths"]["logs"])

    df = pd.read_csv(cfg["paths"]["metadata"])
    raw_dir = Path(cfg["paths"]["raw_videos"])
    fps = cfg["video"]["target_fps"]
    res = cfg["video"]["resolution"]
    sr = cfg["audio"]["sample_rate"]

    done = missing = failed = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Stage 1 preprocessing"):
        vid = row["video_id"]
        raw = raw_dir / f"{vid}{RAW_EXT}"
        if not raw.exists():
            missing += 1
            continue
        try:
            if not Path(row["video_path"]).exists():
                normalize_video(str(raw), row["video_path"], fps=fps, resolution=res)
            if not Path(row["audio_path"]).exists():
                extract_audio(str(raw), row["audio_path"], sample_rate=sr)
            done += 1
        except Exception as e:  # noqa: BLE001
            failed += 1
            log.error(f"{vid}: {e}")

    log.info(f"Stage 1 complete | processed={done} missing_raw={missing} failed={failed}")
    print(f"processed={done}  missing_raw={missing}  failed={failed}")


if __name__ == "__main__":
    run()
