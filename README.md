# COCO Object Detection: YOLO vs Faster R-CNN (Constrained Compute)

A video recording of the presentation of this project can be found at [Youtube](https://www.youtube.com/watch?v=Ox1L8beEo-4) 

This repository compares YOLO and Faster R-CNN on COCO-style data under strict single-GPU constraints (RTX 3050 Ti 4GB), including:

- Short-schedule ablations (10 epochs)
- Extended hero continuation (30 epochs total)
- Held-out non-sampled test evaluation
- Quantitative and qualitative before/after analysis

## Project Goals

1. Compare YOLO and Faster R-CNN under limited VRAM and runtime.
2. Measure sensitivity to backbone (ResNet-18 vs ResNet-50) and optimizer (SGD vs Adam).
3. Test whether longer training changes family ranking.

## Headline Results

### Baseline (10 Epochs, from runs/summary_results.csv)

- Best YOLO config: ResNet-50 + SGD
  - AP 0.3639, AP50 0.5122, AP75 0.3830, train 812.21s
- Best Faster R-CNN config: ResNet-50 + SGD
  - AP 0.1189, AP50 0.2531, AP75 0.0943, train 1538.13s

### Held-out Non-Sampled Test (Before vs After Hero)

From runs/qualitative_hero/hero_test_eval_metrics.csv:

- YOLO
  - Before: AP 0.4607, AP50 0.6267, AP75 0.4963
  - After:  AP 0.3634, AP50 0.5206, AP75 0.3894
- Faster R-CNN
  - Before: AP 0.1106, AP50 0.2448, AP75 0.0798
  - After:  AP 0.2270, AP50 0.4068, AP75 0.2345

Interpretation:

- Faster R-CNN improves substantially with extended training.
- YOLO degrades relative to its earlier checkpoint in this hero continuation.
- Ranking does not flip: YOLO remains best by AP on held-out test.

## Repository Layout

- comparison.ipynb
  - Main experimental notebook for baseline runs and summary artifacts.
- hero_qualitative_extension.ipynb
  - Hero continuation notebook with before/after qualitative panels and held-out test before/after metrics.
- results_presentation.ipynb
  - Clean presentation notebook for HTML export and video walkthrough.
- neurips_report.tex
  - NeurIPS-style report manuscript.
- neurips_report_draft.md
  - Markdown report draft.
- eval_nonsampled_test.py
  - Script version of held-out non-sampled test evaluation.
- runs/
  - All generated artifacts, checkpoints, metrics, and figures.

## Key Artifacts

- runs/summary_results.csv
- runs/comparison_plots.png
- runs/qualitative_hero/hero_test_eval_metrics.csv
- runs/qualitative_hero/hero_test_eval_deltas.csv
- runs/qualitative_hero/hero_test_ap_deltas.png
- runs/qualitative_hero/hero_test_before_after_ap.png
- runs/qualitative_hero/hero_test_before_after_ap50.png
- runs/qualitative_hero/hero_test_before_after_ap75.png
- runs/qualitative_hero/panels/
- runs/detect/runs/yolo_resnet50_sgd_hero/results.png

## Environment

Recommended environment:

- Python 3.12 (conda environment used in this project: cogs185)
- PyTorch + torchvision
- Ultralytics
- pycocotools
- pandas, matplotlib, Pillow

Example setup:

```bash
conda create -n cogs185 python=3.12 -y
conda activate cogs185
pip install torch torchvision ultralytics pycocotools pandas matplotlib pillow
```

## How to Reproduce

### 1) Baseline Experiments

Run notebook:

- comparison.ipynb

Expected outputs include:

- runs/summary_results.csv
- runs/comparison_plots.png

### 2) Hero Continuation + Before/After Analysis

Run notebook:

- hero_qualitative_extension.ipynb

This notebook:

1. Loads best checkpoints.
2. Saves before qualitative predictions.
3. Continues hero training to 30 epochs (checkpoint-safe).
4. Saves after qualitative predictions.
5. Evaluates before/after on held-out non-sampled test and saves delta plots.

### 3) Presentation Notebook (for HTML/video)

Run notebook:

- results_presentation.ipynb

Then export to HTML from Jupyter or VS Code.

## Data Protocol Notes

- Training subset is sampled from local COCO 2017 exports.
- Held-out test is sampled from the non-sampled train pool for leakage control.
- Test split info is stored at:
  - runs/qualitative_hero/hero_test_split_info.json

## Common Issues

- CUDA device-side assert after a failed run:
  - Restart kernel and rerun setup cells.
- Missing pycocotools:
  - Install pycocotools in the active Python environment.
- YOLO resume errors at completed epoch count:
  - Use hero notebook logic that skips YOLO training when already complete and still runs Faster R-CNN continuation.
