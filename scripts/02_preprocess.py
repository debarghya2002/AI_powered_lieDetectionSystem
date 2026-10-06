"""Step 2: normalize video (.wmv->.mp4) + extract 16kHz mono audio."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from preprocessing.run_preprocessing import run

if __name__ == "__main__":
    run("configs/default.yaml")
