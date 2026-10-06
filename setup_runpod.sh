#!/usr/bin/env bash
# Runpod (Linux GPU) setup via Miniconda. Same env as Windows; CUDA torch auto-selected.
set -e
conda env create -f environment.yml || conda env update -f environment.yml
echo "Done. Run: conda activate liedetect"