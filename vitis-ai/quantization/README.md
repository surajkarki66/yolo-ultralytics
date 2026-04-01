# Vitis AI Quantization

This directory contains scripts and data for quantizing YOLOv8 models for deployment on AMD Vitis AI (e.g. DPUCZDX8G). Ultralytics and PyTorch are assumed to be installed.

## Directory layout

- `custom_layers/` — Vitis AI–compatible layer definitions (`block.py`, `head.py`) used during quantization.
- `data/` — Calibration dataset for quantization; see `data/README.md` for the expected structure.
- `patch_ultralytics.py` — Patches the installed Ultralytics package so models can be loaded and quantized correctly.
- `prepare_calibration_data.py` — Prepares calibration data from your dataset.
- `run_compression.sh` — Runs the full quantization/compression pipeline.
- `run_yolo_conversion.sh` — Runs the SiLU → HardSwish model conversion only.
- `vai_q_yolo.py` — Main Vitis AI quantization script for YOLO.
- `yolo_converter.py` — Converts Ultralytics YOLO weights (SiLU → HardSwish) for Vitis AI compatibility.

## Procedure

### Step 1: Model layer conversion (SiLU → HardSwish)

SiLU layers are not supported by Vitis AI; they are replaced with HardSwish, which is very similar, so retraining is typically not required.

Run the converter:

```bash
python yolo_converter.py <path_to_best.pt> <output_dir> <model_name>
```

- `best.pt` — Path to your Ultralytics weights file (e.g. `~/best.pt`).
- `output_dir` — Directory where the converted `.pt` model will be saved.
- `model_name` — Name of the output model (no extension).

Example:

```bash
python yolo_converter.py ./yolov8n_416.pt ./ yolov8n_416_hardswish
```

> **Note:** Ultralytics `.pt` files contain the model, weights, and other parameters. The converter extracts the model, replaces SiLU with HardSwish, and saves it so it can be used with Vitis AI’s PyTorch flow (e.g. `torch.jit.trace`).

### Step 2: Patch Ultralytics (inside Vitis AI container)

In the Vitis AI Docker environment:

1. Activate the PyTorch environment: `conda activate vitis-ai-pytorch`
2. Install Ultralytics: `pip install ultralytics==8.1.47`
3. Apply the patch before running quantization/compilation: `python patch_ultralytics.py`

The patch updates model class definitions so the (HardSwish-converted) models load and trace correctly. After conversion we use the DetectionModel backend rather than the high-level YOLO class; behavior is equivalent.

### Step 3: Run quantization

```bash
python vai_q_yolo.py --model_name <path_to_converted_model.pt> --batch_size <batch_size> --target DPUCZDX8G_ISA1_B4096 --quant_mode <calib|test>
```

- `model_name` — Path to the **converted** `.pt` from Step 1 (e.g. `yolov8n_416_hardswish.pt`).
- `batch_size` — Use a small value (e.g. 16 or 32) to avoid OOM.
- `quant_mode` — `calib` for calibration (run first); `test` for evaluation and deployment artifacts.

Run calibration once, then use `test` for deployment.

## Dependencies

See `requirements.txt` in this directory. Use the Vitis AI Docker image and `vitis-ai-pytorch` conda environment for compatibility.
