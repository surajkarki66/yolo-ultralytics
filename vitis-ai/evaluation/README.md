# Vitis AI evaluation

This package evaluates quantized and/or compiled YOLO models (Vitis AI flow) for **object detection (Detect)** and **oriented bounding boxes (OBB)**. Dependencies are managed with `pyproject.toml` and `uv.lock` in this directory.

## Structure

- **Detect/** — Evaluation for standard YOLOv8 detection (axis-aligned boxes).
  - `coco_prep.py` — Prepares COCO-format data/labels.
  - `evaluate.py` — Runs evaluation on the quantized/compiled detection model.
  - `post_processing.py` — Post-processing of DPU outputs (e.g. NMS, decoding).
  - `utils.py` — Shared utilities.
  - `visualize_output.py` — Visualizes detection results.
  - `models/` — Place quantized/compiled detection models here; see `Detect/models/README.md`.
  - `test_data/` — Test images and labels; see `Detect/test_data/README.md`.

- **OBB/** — Evaluation for YOLOv8 OBB (oriented bounding boxes).
  - `evaluate.py` — Runs evaluation for OBB models.
  - `post_processing.py` — OBB-specific post-processing.
  - `remove_dfl.py` — Removes DFL (Distribution Focal Loss) layer for deployment.
  - `test_data/` — OBB test data; see `OBB/test_data/README.md`.

## Usage

From this directory (or with this directory on `PYTHONPATH`), run the appropriate `evaluate.py` under `Detect/` or `OBB/` after placing models and test data as described in the subdirectory READMEs. Post-processing is applied to the DPU output before computing metrics.
