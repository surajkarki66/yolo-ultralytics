# Vitis AI quantization

Prepare and quantize YOLOv8 Detect models for AMD DPU targets using `pytorch_nndct`.

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

## Environment

```bash
conda activate vitis-ai-pytorch   # Vitis AI Docker / conda env
cd vitis-ai/quantization
pip install -r requirements.txt   # pins ultralytics==8.1.47
python patch_ultralytics.py
```

Train or export a `.pt` checkpoint in the main repo first (`python main.py train`), then copy `best.pt` (or your checkpoint) into this directory or pass an absolute path.

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
- ONNX / TorchScript intermediates (depending on Vitis version)

Copy artifacts to:

- `../compilation/model/` — for `vai_c_xir`
- `../evaluation/Detect/` or `../evaluation/OBB/` — pass the `.pkl` to post-processing scripts

## Notes

- Checkpoint must be Ultralytics format: `torch.load(path)["model"]`.
- Detect head export assumes a standard YOLOv8 `Detect` head at `model.model[-1]`.
- For multiple DPU runs, rename `quantize_result` after each `run_compression.sh` block to avoid overwrites.
