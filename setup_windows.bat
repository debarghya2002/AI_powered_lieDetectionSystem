@echo off
REM Windows 11 (CPU) setup via Miniconda
call conda env create -f environment.yml || call conda env update -f environment.yml
echo Done. Run:  conda activate liedetect