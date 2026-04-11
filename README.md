# Efficient YOLO Pipeline for Edge Deployment

This repository provides an end-to-end workflow for YOLO model development and deployment:

- dataset checks and visualization
- training, testing, tuning, benchmark, and export
- Vitis AI quantization, compilation, and evaluation for Detect and OBB pipelines

The main CLI entrypoint is `main.py`.

## Version Compatibility

- This repository can be adapted for multiple YOLO versions in general workflows.
- For the Vitis AI-compatible training and deployment path in this repo, supported models are:
  - `YOLOv26` / `YOLOv11` (Detect)
  - `YOLOv26-OBB` / `YOLOv11-OBB` (Oriented Bounding Boxes)

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
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
- `eda` supports `.jpg`, `.jpeg`, and `.png`.
- images missing label files are still counted by EDA (as zero-annotation images).
- training/testing/tuning/export behavior is configured via `config.yaml`.

## Project Areas

- `dataset/` - dataset layout and label format (YOLO + YOLO OBB)
- `dataset_utils/` - `eda.py` and `visualize_dataset.py`
- `models/yolo/` - train/test/tune/export/benchmark/cross-validation modules
- `vitis-ai/quantization/` - quantization scripts (without yolo_converter flow)
- `vitis-ai/compilation/` - `.xmodel` compilation for target DPU architectures
- `vitis-ai/evaluation/` - Detect and OBB post-processing + evaluation scripts

## Vitis AI High-Level Flow

1. Quantize model in `vitis-ai/quantization/`
2. Compile in `vitis-ai/compilation/`
3. Evaluate in `vitis-ai/evaluation/Detect/` or `vitis-ai/evaluation/OBB/`

> **Note:** FPGA deployment through Vitis AI in this repository requires `ultralytics==8.4.24`.  
> This is a strict dependency for the supported quantization/patching workflow.

## End-to-End Workflow (Three Phases)

### Phase 1: Training Phase

```bash
pip install -r requirements.txt
```

```bash
python patch_ultralytics.py
```

```bash
python3 main.py train
```

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

### Phase 3: Compilation Phase

```bash
cd vitis-ai/compilation
vai_c_xir -x YOLOv26/DetectionModel_int.xmodel -a Architectures/arch_B4096.json -o zynq_output/yolov26n/ -n yolov26n
```

The `YOLOv26/` and `YOLOv26-OBB/` directories are used for **both** YOLOv26 and YOLOv11 quantized `.xmodel` files (same layout and compilation flow).

For OBB, compile the model under `YOLOv26-OBB/` (YOLOv26-OBB or YOLOv11-OBB) with the matching output name.

## Dataset Layout

```text
dataset/
  data.yaml
  train/images  train/labels
  valid/images  valid/labels
  test/images   test/labels
```

For detailed label formats (standard YOLO and YOLO OBB), see `dataset/README.md`.