# esd-video-phase

Video vision-language models for **four-phase recognition in endoscopic submucosal dissection (ESD)**,
trained and tested across two hospitals.

Four video VLMs (Qwen3.8-27B, Molmo2-8B, InternVideo3-8B, VideoChat3-4B) are fine-tuned with the same
LoRA recipe on seven training-data combinations and evaluated on three test sets with one per-second
protocol:

| Source | Cases | Use |
| --- | --- | --- |
| NTUH colorectal ESD (in-house, hand-cut clips) | 13 | case-level 5-fold split; out-of-fold or external |
| Renji ESD for CNLs (colorectal, public) | 30 | case-level 5-fold split; out-of-fold or external |
| Renji ESD for EGC (gastric, public) | 20 | case-level 5-fold split; out-of-fold or external |

Training combinations: NTUH, CNLs, EGC, CNLs+EGC, NTUH+CNLs, NTUH+EGC, NTUH+CNLs+EGC
(fold *k* of a combination is the union of each source's own fold *k*).
Phases: Diagnosis, Local injection, Incision / Dissection, Hemostasis.

No videos, frames, weights or patient data are included. NTUH cases appear only as pseudonyms
(`NTUH_01`–`NTUH_13`) in `configs/splits/ntuh_5fold.tsv`.

## Main result (case-macro-F1, mean of the three test sets)

| Model | NTUH only | CNLs+EGC | NTUH+CNLs+EGC |
| --- | --- | --- | --- |
| Molmo2-8B | 0.699 | 0.819 | **0.848** |
| VideoChat3-4B | 0.601 | 0.784 | 0.831 |
| InternVideo3-8B | 0.648 | 0.798 | 0.822 |
| Qwen3.8-27B | 0.628 | 0.799 | 0.808 |

Per test set, per phase, with confidence intervals and paired tests: `python scripts/score.py`.

## Layout

```
configs/      models.yaml (weights, revisions, batch sizes), splits/ntuh_5fold.tsv
src/esdvideo/
  contracts.py    prompt, JSON answer contract and its parser
  windows.py      video windows (1 fps, <= 32 s), dataset records, per-second scoring
  ntuh.py         NTUH clip enumeration and 1 fps extraction (endoscopic-view crop)
  renji_frames.py Renji labels and first-frame-per-second extraction (Experiment 1 frames)
  renji_clips.py  Renji training clips: monotonic 4-phase fit, pure / boundary clips, 5-fold split
  video.py        clip extraction with ffmpeg (PyAV fallback)
  vlm.py          LoRA training / prediction for Molmo2, InternVideo3, VideoChat3
  qwen.py         LlamaFactory configs for Qwen3.8-27B
  metrics.py      case-macro-F1, frame-micro-F1, per-phase F1
  stats.py        case-level bootstrap CI, paired bootstrap, Wilcoxon
  clip_eval.py    clip-level scorer (pooled macro-F1, exact match, boundary MAE)
scripts/      prepare_*.py (data), run_job.py (train + predict), score.py, hemostasis.py, make_report.py
jobs/         queue.sh (one worker per GPU) and all_jobs.txt (140 jobs)
report/       HTML report template (Traditional Chinese)
tests/        unit tests (windows, answer parsing, metrics)
```

## Setup

Two Python environments are needed, because Qwen3.8 and the other three models require different
`transformers` versions:

* **Qwen environment**: PyTorch 2.11 (CUDA 13), transformers 5.5, PEFT 0.18, LlamaFactory 0.9.5, ffmpeg.
* **VLM environment**: the same PyTorch with `transformers==4.57.3` and `huggingface_hub<1.0`
  (`pip install -e ".[vlm]"`). VideoChat3 runs without flash-attn (see `src/esdvideo/vlm.py`).

```bash
cp paths.example.yaml paths.local.yaml      # edit the six paths
pip install -e ".[test]" && pytest
```

Download the weights listed in `configs/models.yaml` into `<pretrained_root>/<dir>/`, e.g.
`hf download allenai/Molmo2-8B --revision e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b --local-dir <pretrained_root>/Molmo2-8B`.

## Pipeline

```bash
export FFMPEG=ffmpeg
# 1. data
python scripts/prepare_ntuh.py --workers 32                 # in-house data required
python scripts/prepare_renji_frames.py --release cnls
python scripts/prepare_renji_frames.py --release egc
python scripts/prepare_renji_windows.py --release cnls
python scripts/prepare_renji_windows.py --release egc
for k in 0 1 2 3 4; do                                       # Renji training clips, per fold
  python scripts/prepare_renji_clips.py --fold $k --source <renji_cnls_root> \
      --clip-root <work_root>/renji_clips/cache/cnls --output <work_root>/renji_clips/cnls/fold$k
  python scripts/prepare_renji_clips.py --fold $k --source "<renji_egc_root>" \
      --clip-root <work_root>/renji_clips/cache/egc --output <work_root>/renji_clips/egc/fold$k
  python scripts/prepare_renji_clips.py --fold $k --source cnl=<renji_cnls_root> --source "egc=<renji_egc_root>" \
      --clip-root cnl=<work_root>/renji_clips/cache/cnls --clip-root egc=<work_root>/renji_clips/cache/egc \
      --output <work_root>/renji_clips/combined/fold$k
done
python scripts/prepare_training_sets.py

# 2. train + predict (one job = training set x model x fold on one GPU)
export QWEN_BIN=/path/to/qwen-env/bin VLM_PYTHON=/path/to/vlm-env/bin/python
python scripts/run_job.py --set ntuh_cnls --model molmo2 --fold 0 --gpu 0
bash jobs/queue.sh 0 & bash jobs/queue.sh 1 &               # or run all 140 jobs on two GPUs

# 3. evaluation
python scripts/score.py --out results/scores.json --tables results/tables.md
python scripts/hemostasis.py --out results/hemostasis.json
python scripts/make_report.py --scores results/scores.json --out results/report.html
```

## Protocol in brief

* **Input**: 1 fps video windows of at most 32 frames, letterboxed to 512x448, at most 50,176 pixels
  per frame (Molmo2: its fixed 378x378). NTUH frames are cropped to the endoscopic view (right
  1280x1080 of the 1920x1080 monitor capture).
* **Output**: JSON with one or two ordered segments covering the window
  (`{"segments":[{"phase", "start_seconds", "end_seconds"}]}`); invalid answers count as wrong.
* **Training**: LoRA r 8 / alpha 16 on all language-model linear layers, frozen vision encoder,
  trained projector, lr 1e-4 cosine (10 % warmup), 3 epochs, effective batch 6, bf16, epoch with the
  lowest validation loss. NTUH training windows: 32 s, stride 8 s, at most 200 per phase. Renji training
  clips: 32 s single-phase and 64 s transition clips from a monotonic 4-phase fit of the fine labels.
* **Evaluation**: every second scored once. NTUH: windows tiling the hand-cut clips. Renji: windows
  tiling the whole annotated timeline with the Experiment 1 label mapping (estimation, injection,
  incision/esd, vessel_treatment; preparation, marking and clips excluded). Out-of-fold when the test
  source is in the training data, otherwise external with per-case scores averaged over the five fold
  models. Metrics: case-macro-F1, frame-micro-F1, per-phase F1, case-level bootstrap CIs, paired
  bootstrap and Wilcoxon tests against the single-source model.

## Reproducibility check

This repository is a cleaned-up version of the scripts used for the experiments. It was checked against
the original runs: rebuilding the NTUH and Renji evaluation windows gives byte-identical datasets
(56 files), the Renji 5-fold splits and clip selections are identical, the mixed training sets are
identical, and `scripts/score.py` on the original predictions reproduces all 1,064 reported numbers
exactly. Single seed (20261022); results will vary slightly across hardware and library versions.

## Data

* NTUH ESD clips are in-house data and cannot be shared.
* Renji ESD datasets: the public "ESD for CNLs" and "ESD for EGC" releases (CNLs: Scientific Data 2025,
  12:1366). Please cite the original publications when using them.
