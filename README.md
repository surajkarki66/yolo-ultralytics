# Efficient YOLO Pipeline for Edge Deployment

This repository provides an end-to-end workflow for YOLO model development and deployment:

- dataset checks and visualization
- training, testing, tuning, benchmark, and export
- Vitis AI inspection, quantization, compilation, evaluation, and on-target inference for Detect and OBB pipelines

The main CLI entrypoint is `main.py`.

## Version Compatibility

- This repository can be adapted for multiple YOLO versions in general workflows.
- For the Vitis AI-compatible training and deployment path in this repo, supported models are:
  - `YOLOv26` / `YOLOv11` (Detect)
  - `YOLOv26-OBB` / `YOLOv11-OBB` (Oriented Bounding Boxes)

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

## Vitis AI High-Level Flow

1. Inspect model in `vitis-ai/inspection/` (optional, recommended before quantization)
2. Quantize model in `vitis-ai/quantization/`
3. Compile in `vitis-ai/compilation/`
4. Evaluate in `vitis-ai/evaluation/Quantized_Model/` or `vitis-ai/evaluation/Compiled_Model/`
5. Deploy (optional) in `vitis-ai/inference/` (YOLOv26 or YOLOv11 paths)

> **Note:** FPGA deployment through Vitis AI in this repository requires `ultralytics==8.4.24`.  
> This is a strict dependency for the supported quantization/patching workflow.

## End-to-End Workflow

### Phase 1: Training (host PC)

```bash
pip install -r requirements.txt   # or: uv sync
python patch_ultralytics.py       # patches custom_layers/conv.py and block.py into Ultralytics
python3 main.py train
```

Edit **`config.yaml`** first (`training.data`, `testing.model_path`, etc.). Commands are implemented under `scripts/` and invoked by `main.py`.

### Phase 1b: Inspection (optional, inside Vitis AI)

Validate DPU compatibility before quantization:

```bash
conda activate vitis-ai-pytorch
cd vitis-ai/quantization && python patch_ultralytics.py
cd ../inspection
python inspection.py --model_path best.pt --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B4096
```

See `vitis-ai/inspection/README.md` for details and `inspect.sh` for per-DPU examples.

### Phase 2: Quantization Phase (inside Vitis AI)

Activate the Vitis AI PyTorch environment first:

```bash
conda activate vitis-ai-pytorch
```

Install quantization dependencies:

```bash
cd vitis-ai/quantization
pip install -r requirements.txt
```

Patch Ultralytics in the Vitis AI environment:

```bash
cd vitis-ai/quantization
python patch_ultralytics.py
```

Prepare calibration data:

```bash
cd vitis-ai/quantization
python prepare_calibration_data.py
```

Run quantization (calib then test). Use `--model_path` for your trained or converted `.pt` (match `--img_height` / `--img_width` to your model, e.g. 416):

```bash
cd vitis-ai/quantization
python vai_q_yolo.py --model_path <trained_or_exported_model.pt> --batch_size 16 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B4096 --quant_mode calib
python vai_q_yolo.py --model_path <trained_or_exported_model.pt> --batch_size 1 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B4096 --quant_mode test --deploy
```

This generates quantized artifacts (including `.xmodel`) for deployment.

Details: [vitis-ai/quantization/README.md](vitis-ai/quantization/README.md)

### Phase 3: Compilation Phase

Copy the quantized `.xmodel` into `vitis-ai/compilation/model/`, then compile:

```bash
cd vitis-ai/compilation
vai_c_xir -x model/DetectionModel_int.xmodel -a Architectures/arch_B4096.json -o zynq_output/yolov26n/ -n yolov26n
```

Place both model types in `model/`:

- `DetectionModel_int.xmodel` — YOLOv26 / YOLOv11 detection
- `OBBModel_int.xmodel` — YOLOv26-OBB / YOLOv11-OBB

For OBB, use `model/OBBModel_int.xmodel` with the matching output name (e.g. `yolov26n_obb`). See `vitis-ai/compilation/run_compile.sh` for all architecture options.

Details: [vitis-ai/compilation/README.md](vitis-ai/compilation/README.md)

### Phase 4: Evaluation

| Task | Path | Script |
|------|------|--------|
| Detect / OBB (ONNX) | `vitis-ai/evaluation/Quantized_Model/` | `eval_onnx.py` |
| Detect (FPGA NPZ) | `vitis-ai/evaluation/Compiled_Model/` | `fpga_inference.py` → `eval_predictions_npz.py` |
| OBB (FPGA NPZ) | `vitis-ai/evaluation/Compiled_Model/` | `fpga_inference_obb.py` → `eval_predictions_npz.py` |

Pass the `*_config_no_srd_reg_nc_dfl.pkl` from quantization via `--quant-meta`.

Details: [vitis-ai/evaluation/README.md](vitis-ai/evaluation/README.md)

### Phase 5: On-target inference (optional)

Copy compiled `.xmodel` and config pickle to `vitis-ai/inference/**/model/`.

| Variant | Path |
|---------|------|
| YOLOv26 detect / OBB video | `vitis-ai/inference/YOLOv26/` |
| YOLOv11 C++ FPS + video | `vitis-ai/inference/YOLOv11/` |

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