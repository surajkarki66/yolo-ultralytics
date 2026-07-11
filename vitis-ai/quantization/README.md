# Vitis AI quantization

Prepare and quantize **YOLOv26** and **YOLOv11** models (Detect and OBB) for AMD DPU targets using `pytorch_nndct`.

OBB models use the same script when the checkpoint uses the patched OBB head; see commented blocks in `run_compression.sh`.

## Contents

| File / folder | Role |
|---------------|------|
| `vai_q_yolo.py` | Calibration (`calib`), quantized test (`test`), optional `--deploy` xmodel export |
| `patch_ultralytics.py` | Copies `custom_layers/` into installed Ultralytics `nn/modules` |
| `custom_layers/` | `block.py`, `head.py` for DPU-friendly modules |
| `prepare_calibration_data.py` | Builds `data/val_ids.txt` from `data/val/` |
| `run_compression.sh` | Example calib + test + deploy commands per DPU size |
| `data/` | Calibration images ([data/README.md](data/README.md)) |

## Environment (Vitis AI Docker)

Use the official [Xilinx Vitis-AI](https://github.com/Xilinx/Vitis-AI) Docker image for quantization and compilation.

**Quick start (from repo root):**

```bash
# 1. Place calibration images in vitis-ai/quantization/data/ (or data/val/)
# 2. Copy best.pt into vitis-ai/quantization/
bash vitis-ai/quantization/run_in_vitis_docker.sh
```

**Manual start via official `docker_run.sh`:**

```bash
git clone https://github.com/Xilinx/Vitis-AI.git
cd "/path/to/yolov26-ultralytics"
/path/to/Vitis-AI/docker_run.sh xilinx/vitis-ai-pytorch-cpu:latest \
  bash vitis-ai/quantization/run_yolov26_end2end_pipeline.sh
```

Inside the container, conda env `vitis-ai-pytorch` is activated automatically by the pipeline script.

**Local conda (if already installed):**

```bash
conda activate vitis-ai-pytorch
cd vitis-ai/quantization
pip install -r requirements.txt   # pins ultralytics==8.4.24
python patch_ultralytics.py
```

Train a `.pt` checkpoint in the main repo first (`python main.py train`), then copy `best.pt` (or your checkpoint) into this directory or pass an absolute path.

Optionally run **`../inspection/inspection.py`** first to verify DPU compatibility for your target (see `../inspection/README.md`).

## Calibration data

```bash
# Place images in data/val/, then:
python prepare_calibration_data.py
```

`vai_q_yolo.py` reads image paths from `data/val_ids.txt`, resizes to `--img_height` × `--img_width`, and normalizes to `[0, 1]`.

## Quantization commands

**Calibration** (collects stats, writes config pickle):

```bash
python vai_q_yolo.py \
  --model_path best.pt \
  --batch_size 16 \
  --img_height 416 \
  --img_width 416 \
  --target DPUCZDX8G_ISA1_B4096 \
  --quant_mode calib
```

**Test + deploy** (exports xmodel; use `batch_size 1` for deploy):

```bash
python vai_q_yolo.py \
  --model_path best.pt \
  --batch_size 1 \
  --img_height 416 \
  --img_width 416 \
  --target DPUCZDX8G_ISA1_B4096 \
  --quant_mode test \
  --deploy
```

Or use `run_compression.sh` and uncomment the DPU block you need.

## CLI options (`vai_q_yolo.py`)

| Option | Description |
|--------|-------------|
| `--model_path` | Ultralytics checkpoint (`.pt`) with `["model"]` key |
| `--quant_mode` | `float`, `calib`, or `test` |
| `--deploy` | Export xmodel (only in `test` mode; forces batch size 1) |
| `--target` | DPU target, e.g. `DPUCZDX8G_ISA1_B4096` |
| `--batch_size` | Calibration / inference batch size |
| `--img_height`, `--img_width` | Input size (must match training/deployment) |
| `--config_file` | Optional NNDCT quant config YAML |

## Outputs (`quantize_result/`)

After calibration:

- `{stem}_config_no_srd_reg_nc_dfl.pkl` — head metadata (`no`, stride, `reg_max`, `nc`, DFL) for post-processing  
  Example: `best.pt` → `best_config_no_srd_reg_nc_dfl.pkl`

After test/deploy:

- `DetectionModel_int.xmodel` (typical detect export name)
- `OBBModel_int.xmodel` (typical OBB export name)
- ONNX / TorchScript intermediates (depending on Vitis version)

Copy artifacts to:

- `../compilation/model/` — place `.xmodel` files for `vai_c_xir` (see `../compilation/README.md`)
- `../evaluation/Quantized_Model/` or `../evaluation/Compiled_Model/` — pass the `.pkl` to post-processing scripts

## YOLOv26 end2end (raw DPU export)

YOLOv26 uses a **one-to-one end2end head** (`end2end=True`, `reg_max=1`). The DPU exports **raw one2one logits** (3 outputs per FPN level). Decode and post-processing (top-k or NMS) run on the CPU.

```bash
python vai_q_yolo.py \
  --model_path best.pt \
  --quant_mode calib \
  --target DPUCZDX8G_ISA1_B4096 \
  --img_height 416 --img_width 416

python vai_q_yolo.py \
  --model_path best.pt \
  --quant_mode test --deploy \
  --batch_size 1 \
  --target DPUCZDX8G_ISA1_B4096 \
  --img_height 416 --img_width 416
```

The calib pickle stores the `end2end` flag for evaluation scripts. Pass it via `--quant-meta` in `../evaluation/`.

- **Default** (no flag): one2many export, CPU NMS post-processing.
- **`--end2end`**: fuse one2many away, export one2one, CPU top-k post-processing.

## Notes

- Checkpoint must be Ultralytics format: `torch.load(path)["model"]`.
- Head export assumes the detect/OBB head at `model.model[-1]` (patched via `custom_layers/head.py`).
- Match `--img_height` and `--img_width` to your training and deployment input size (e.g. 416).
- For multiple DPU runs, rename `quantize_result` after each `run_compression.sh` block to avoid overwrites.
