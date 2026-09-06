# Efficient YOLO Pipeline for Edge Deployment

This repository provides an end-to-end workflow for YOLO model development and deployment:

- dataset checks and visualization
- training, testing, tuning, benchmark, and export
- Vitis AI inspection, quantization, compilation, evaluation, and on-target inference for Detect and OBB pipelines

The main CLI entrypoint is `main.py`.

## Version Compatibility

- This repository can be adapted for multiple YOLO versions in general workflows.
- For the Vitis AI-compatible training and deployment path in this repo, supported models are:
  - `YOLOv8` (Detect)
  - `YOLOv8-OBB` (Oriented Bounding Boxes)

## Quick Start

Requires **Python 3.12+** (see `.python-version` / `pyproject.toml`).

**Option A — venv + pip:**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Option B — uv:**

```bash
uv sync
source .venv/bin/activate
```

## Main CLI

```bash
python3 main.py eda --data-dir dataset --output-dir output
python3 main.py visualize --data-dir dataset --output-dir output --num-samples 10
python3 main.py train
python3 main.py test
python3 main.py cross-validation
python3 main.py tune
python3 main.py benchmark
python3 main.py export
```

## CLI Notes

- `visualize --num-samples` is applied per split (`train`, `valid`, `test`).
- `eda` and `visualize` support `.jpg`, `.jpeg`, and `.png` (case-insensitive).
- EDA handles YOLO bbox (5 fields/line) and OBB (9 fields/line) labels; images without label files count as zero annotations.
- Training, testing, tuning, export, benchmark, and cross-validation read **`config.yaml`** — update dataset and checkpoint paths before running (defaults are placeholders).

## Project Areas

- `dataset/` — dataset layout and label format (YOLO + YOLO OBB)
- `dataset_utils/` — EDA and visualization (`eda.py`, `visualize_dataset.py`)
- `custom_layers/` — DPU-friendly `conv.py` / `block.py` patched into Ultralytics (root `patch_ultralytics.py`)
- `scripts/` — train, test, tune, export, benchmark, cross-validation (via `main.py`)
- `config.yaml` — paths and hyperparameters for `main.py` commands
- `vitis-ai/` — inspection, quantization, compilation, evaluation, and on-target inference  
  Start here: [vitis-ai/README.md](vitis-ai/README.md)

## Vitis AI high-level flow

1. **Inspect** (optional) — `vitis-ai/inspection/` (NNDCT `Inspector`)
2. **Quantize** — `vitis-ai/quantization/` (`vai_q_yolo.py`)
3. **Compile** — `vitis-ai/compilation/` (`model/` + `vai_c_xir`)
4. **Evaluate** — `vitis-ai/evaluation/` (Detect or OBB, Quantized or Compiled path)
5. **Deploy** (optional) — `vitis-ai/inference/` (C++ FPS, realtime video)

> **Note:** The Vitis quantization workflow uses **`ultralytics==8.1.47`** (`vitis-ai/quantization/requirements.txt`).  
> Root training uses the version in `requirements.txt`. Patch Ultralytics separately in each environment (see below).

## End-to-End Workflow

### Phase 1: Training (host PC)

```bash
pip install -r requirements.txt   # or: uv sync
python patch_ultralytics.py       # patches custom_layers/conv.py and block.py into Ultralytics
python3 main.py train
```

Edit **`config.yaml`** first (`training.data`, `testing.model_path`, etc.). Commands are implemented under `scripts/` and invoked by `main.py`.

### Phase 1b: Inspection (optional, Vitis AI environment)

Validate DPU compatibility before quantization:

```bash
conda activate vitis-ai-pytorch
cd vitis-ai/quantization && python patch_ultralytics.py
cd ../inspection
python inspection.py --model_path best.pt --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B4096
```

See `vitis-ai/inspection/README.md` for details and `inspect.sh` for per-DPU examples.

### Phase 2: Quantization (Vitis AI environment)

```bash
conda activate vitis-ai-pytorch
cd vitis-ai/quantization
pip install -r requirements.txt
python patch_ultralytics.py          # copies quantization/custom_layers/ into Ultralytics
python prepare_calibration_data.py   # needs images in data/val/
```

Calibration then deploy export:

```bash
python vai_q_yolo.py --model_path best.pt --batch_size 16 --img_height 416 --img_width 416 \
  --target DPUCZDX8G_ISA1_B4096 --quant_mode calib
python vai_q_yolo.py --model_path best.pt --batch_size 1 --img_height 416 --img_width 416 \
  --target DPUCZDX8G_ISA1_B4096 --quant_mode test --deploy
```

Outputs in `quantize_result/`:

- `DetectionModel_int.xmodel` (typical detect name)
- `best_config_no_srd_reg_nc_dfl.pkl` (stem from `--model_path`)

Details: [vitis-ai/quantization/README.md](vitis-ai/quantization/README.md)

### Phase 3: Compilation

Copy `.xmodel` files into `vitis-ai/compilation/model/`, then uncomment your DPU block in `run_compile.sh` or run:

```bash
cd vitis-ai/compilation
vai_c_xir -x model/DetectionModel_int.xmodel -a DPUCZDX8G/arch_B4096.json -o zynq_output/yolov8n/ -n yolov8n
```

OBB: `model/OBBModel_int.xmodel` → `zynq_output/yolov8n_obb/`.

Details: [vitis-ai/compilation/README.md](vitis-ai/compilation/README.md)

### Phase 4: Evaluation

| Task | Folder |
|------|--------|
| Detect (ONNX) | `vitis-ai/evaluation/Detect/Quantized_Model/` |
| Detect (FPGA NPZ) | `vitis-ai/evaluation/Detect/Compiled_Model/` |
| OBB (ONNX) | `vitis-ai/evaluation/OBB/Quantized_Model/` |
| OBB (FPGA NPZ) | `vitis-ai/evaluation/OBB/Compiled_Model/` |

Pass the `*_config_no_srd_reg_nc_dfl.pkl` from quantization to post-processing scripts.

Details: [vitis-ai/evaluation/README.md](vitis-ai/evaluation/README.md)

### Phase 5: On-target inference (optional)

Copy compiled `.xmodel` and config pickle to `vitis-ai/inference/**/model/`.

Details: [vitis-ai/inference/README.md](vitis-ai/inference/README.md)

## Dataset Layout

```text
dataset/
  data.yaml
  train/images  train/labels
  valid/images  valid/labels
  test/images   test/labels
```

For detailed label formats (standard YOLO and YOLO OBB), see `dataset/README.md`.