"""Step 1: codebook -> data/master_metadata.csv."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from datasets.metadata_builder import build_metadata

if __name__ == "__main__":
    build_metadata("data/MU3D_Codebook.xlsx", "data/master_metadata.csv")
