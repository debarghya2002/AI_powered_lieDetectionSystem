# MU3D Multimodal Deception Detection

A CPU-first multimodal lie-detection system built on the **MU3D** (Miami
University Deception Detection Database). The model fuses facial and audio
behavior with a temporal cross-modal transformer, regularizes with a
multi-task auxiliary head, removes subject bias via adversarial domain
adaptation, and is evaluated under a strict Leave-One-Subject-Out (LOSO)
protocol. A Streamlit demo lets you run the trained model on any single video
file.

> **Status: FINAL — result accepted.** All 9 evaluation stages are
> implemented and evaluated. Accuracy 0.575, ROC-AUC 0.594 on the full
> 80-fold LOSO — above chance (0.50) and above published human
> lie-detection accuracy (~54%). Stage 10 (single-video demo) added on top,
> prediction-only, no retraining. Full writeup: `result_mu3d.docx`.

---

## 1. Project objective

Given one full video of a single speaker, predict whether the statement is a
**lie** or a **truth**, with a calibrated probability. One full video = one
sample (no segmentation).

Pipeline at a glance (current architecture):

```
VideoMAE  (8-step temporal sequence)   ┐
Landmarks (16-frame raw sequence)      │
wav2vec2  (16-segment temporal seq.)   ├─► ONE temporal transformer ─► heads
Prosody   (16-segment × 7 contour)     │        (~57 tokens incl. CLS)
                                       ┘        │
                                                ├─ lie head (primary)
                                                ├─ anxiety head (aux)
                                                └─ GRL ─► subject (adversarial)
```

All four modalities are fed as raw per-timestep tokens directly into a single
transformer — no separate facial-fusion/audio-fusion step, no Bi-LSTM
pre-aggregation.

---

## 2. Dataset: MU3D

- **320 videos**, **80 targets** (subjects), **4 videos per target**.
- Conditions, fully crossed: `1PT` positive truth, `2NL` negative lie,
  `3NT` negative truth, `4PL` positive lie.
- `VideoID` prefix encodes the subject, e.g. `BF001`. `subject_id` is derived
  from this and is the LOSO grouping key.
- **Perfectly balanced**: 160 lie/160 truth, 160 neg/160 pos valence (fully
  crossed with veracity), 40F/40M, 40 Black/40 White, 20 per Sex×Race cell.
  → no class imbalance, plain BCE is sufficient.

### Codebook variable roles

| Variable | Type | Role in the model |
|---|---|---|
| `Veracity` | binary | **Primary label** |
| `Anxious` | continuous 1–7 | **Auxiliary target** — anxiety regression head |
| `Valence` | binary | Optional auxiliary head + fold stratification |
| `Attractive`, `Trustworthy` | continuous 1–7 | Optional auxiliary heads (unused) |
| `subject_id` | categorical | LOSO grouping **and** adversarial (GRL) target |
| `Sex`, `Race` | binary | Adversarial heads exist, **off** in final config |
| `Age` | int | Evaluation slice only |
| `VidLength_sec`, `WordCount` | continuous | Normalization/weighting |
| `Accuracy` | continuous | **Weighting only — NOT a feature (leaks, r=0.46)** |
| `TruthProp`, `Sensitivity`, `Criterion` | continuous | **Excluded — leak the label** |

### Video format note

Source `.wmv` files are actually MP4 containers — "wmv→mp4" is usually a fast
remux, not a transcode.

### Note on this copy of the repository

Raw source videos (`data/raw/`) and most of the normalized intermediate
files (`data/processed/`) have been removed from this copy to keep upload
size down — only a small sample remains. This does **not** affect
reproducibility of the reported result: the full cached feature set (all 320
clips, both modalities) is present under `embeddings/` and is what training
and evaluation actually read. Regenerate `data/processed/` from the original
raw videos via Steps 1–2 in the execution manual only if you need to
re-extract embeddings from scratch.

---

## 3. Design decisions log

1. **One video = one sample.** No segmentation; full temporal sequence preserved.
2. **Label** = `Veracity`. Balanced dataset → plain BCE.
3. **Evaluation** = LOSO by `subject_id`. Predictions pooled across all 80
   folds and scored once (no per-fold std-dev).
4. **Multi-task heads:** Lie (primary, BCE) + Anxiety (auxiliary, regression,
   weight ≈0.2) — anxiety chosen for being orthogonal to the label (r=-0.04),
   a genuine regularizer rather than a shortcut.
5. **Domain adaptation** via GRL on the fused representation, adversarial
   classifier on `subject_id` (λ=0.5). **Confirmed essential by ablation** —
   the model collapses to chance (AUC 0.50) without it, even with temporal
   features.
6. **Temporal modeling — not a standalone stage.** An originally-planned
   separate Bi-LSTM/attention stage was never built as its own module;
   temporal handling is folded directly into feature extraction (sequences
   preserved instead of pooled) and the fusion transformer's self-attention.
   This reversed an earlier "skip temporal modeling" plan, after the first
   pooled-feature pipeline hit chance-level LOSO (AUC 0.51) and a
   linear-probe diagnostic confirmed pooled features carried no transferable
   signal.
7. **Excluded from model inputs:** `Accuracy`, `TruthProp`, `Sensitivity`,
   `Criterion` (label leakage).
8. **CPU-first.** Entire project runs on CPU (Windows, conda env
   `liedetect`); no GPU used or required for the accepted result.
9. **Single-video demo (Stage 10) is prediction-only** — reuses the trained
   checkpoint and feature pipeline; trains nothing.

### Metadata schema (`data/master_metadata.csv`)

```
video_id, video_path, audio_path, subject_id, label, valence, anxiety,
sex, race, age, duration_sec, word_count, accuracy, transcript
```

---

## 4. Full workflow (current / final)

**Stage 1 — Data ingestion.** Normalize FPS/resolution, `.wmv → .mp4`,
extract 16kHz mono audio. Build `master_metadata.csv`. Prepare LOSO splits.

**Stage 2 — Facial analysis.** VideoMAE (frozen) kept as an 8-step temporal
sequence `(8, 768)`. MediaPipe landmarks kept as a raw 16-frame sequence
`(16, 1404)` — no Bi-LSTM, no fusion step; both go directly to Stage 4.

**Stage 3 — Audio analysis.** wav2vec2 (frozen) binned into a 16-segment
sequence `(16, 768)`. Prosody as a 16-segment × 7-feature contour `(16, 7)`.
No fusion step; both go directly to Stage 4.

**Stage 4 — Multimodal fusion.** All four raw sequences → per-timestep token
construction (~57 tokens incl. CLS) → modality-type + temporal position
embeddings → self-attention across all four streams → pre-norm transformer
(`norm_first=True`) → CLS → classification head. Loss: weighted BCE.

**Stage 5 — Multi-task learning.** Lie (primary) + Anxiety (auxiliary)
heads off the fused CLS representation. Backbones stayed fully frozen
throughout.

**Stage 6 — Domain adaptation.** GRL + subject-classifier (λ=0.5) on the
fused representation. Confirmed essential by ablation.

**Stage 7 — Temporal modeling — not a standalone stage** (see design
decision 6).

**Stage 8 — Calibration.** Temperature scaling, clamped to **[0.5, 5.0]**
(raw fits were unstable on small validation folds). MC-dropout implemented
but not used in the final reported evaluation.

**Stage 9 — Evaluation (LOSO).** 80 folds, no subject leakage, predictions
pooled and scored once: Accuracy, Precision, Recall, F1, ROC-AUC, EER, ECE.

**Stage 10 — Deployment / external video inference.** Given any saved video
file: sample 16 frames, extract audio, generate the same four features,
standardize using a saved inference bundle, run the trained classifier,
show truth/deception probability in a Streamlit UI. **Prediction-only.**

> **Calibration caveat:** the inference bundle uses `T = 1.0`
> (uncalibrated) rather than the fitted per-fold temperature — same trained
> model, but demo probabilities aren't calibrated the way the reported LOSO
> metrics are.

**Execution order:** Ingestion → Facial → Audio → Fusion → Multi-task →
Domain adaptation → Calibration → LOSO evaluation → Deployment/Prediction.

---

## 5. Project structure

```
mu3d_lie_detection/
├── configs/            # YAML: default, model, preprocess, train
├── data/               # raw/ (sample only), processed/ (sample only),
│                       #   MU3D_Codebook.xlsx, master_metadata.csv
├── datasets/           # metadata_builder, mu3d_dataset, loso_splitter
├── preprocessing/      # video_normalizer, audio_extractor, run_preprocessing
├── models/             # videomae_encoder, landmark_extractor, wav2vec_encoder,
│                       #   prosodic_features, deception_model
├── fusion/             # temporal_cross_modal.py
├── multitask/          # heads (lie + anxiety), multitask_loss
├── domain_adaptation/  # grl, domain_classifier
├── calibration/        # temperature_scaling (clamped), mc_dropout (unused in final eval)
├── evaluation/         # metrics, loso_evaluator
├── embeddings/         # cached SEQUENCE features — facial/, audio/
├── trainers/           # base_trainer, multitask_trainer
├── utils/              # config, device, seed, io, logging_utils, video_frames
├── inference/          # file_predictor.py — single-video prediction, no training code
├── outputs/            # loso_predictions.csv, loso_metrics.json, figures/,
│                       #   best.pt, inference_bundle.pt, demo_uploads/,
│                       #   *_DA_on / *_DA_off ablation backups
├── scripts/            # 01_build_metadata, 02_preprocess, 03_extract_embeddings,
│                       #   03_extract_audio, 04_run_loso, create_inference_bundle,
│                       #   demo_predict
├── logs/               # preprocess.log
├── train.py  overfit_test.py  linear_probe.py  sweep_lambda.py
├── compare_da.py  plot_results.py
├── result_mu3d.docx    # results writeup (Word)
├── setup_windows.bat  setup_runpod.sh
├── requirements.txt  environment.yml
└── README.md           # this file
```

---

## 6. Setup

Runs on Windows, CPU only.

```bash
conda create -n liedetect python=3.10
conda activate liedetect
pip install -r requirements.txt
```

Requires `ffmpeg` on PATH. No GPU required — every reported result was
produced on CPU. `streamlit` is only needed for the Stage 10 demo.

## 7. Run

```bash
# one-time setup
python scripts/01_build_metadata.py
python scripts/02_preprocess.py
python scripts/03_extract_embeddings.py
python scripts/03_extract_audio.py

# sanity checks
python overfit_test.py
python train.py

# full evaluation (slow -- budget several hours on CPU)
python scripts/04_run_loso.py --folds 5     # smoke test first
python scripts/04_run_loso.py

# results
python plot_results.py
```

### Stage 10 — single-video prediction demo

```bash
python scripts/create_inference_bundle.py     # packages the trained checkpoint, trains nothing
python -m streamlit run scripts/demo_predict.py
```

---

## Progress

- [x] Stages 1–9 implemented and evaluated
- [x] Diagnosed chance-level result via linear probe; temporal rework fixed it
- [x] Domain-adaptation ablation and lambda sweep (0.5 confirmed optimal)
- [x] Final result accepted — see `result_mu3d.docx`
- [x] Stage 10 — single-video inference demo added (prediction-only)
- [x] Project cleanup — setup scripts, requirements/environment sync, stale
      embedding caches, stray empty file all corrected

**Not done / documented as future work:** eGeMAPS audio features, facial
Action Units (OpenFace), GPU fine-tuning of the backbones (LoRA), a
calibrated (non-1.0-temperature) inference bundle.