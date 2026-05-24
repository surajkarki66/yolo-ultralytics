# Vitis AI deployment pipeline

End-to-end flow for deploying **YOLOv26** / **YOLOv11** (Detect and OBB) on AMD DPU (Zynq/Kria) using Vitis AI:

1. **Inspection** (optional) — validate DPU compatibility before quantization
2. **Quantization** — calibrate and export `.xmodel` (+ config pickle)
3. **Compilation** — build device-ready artifacts with `vai_c_xir`
4. **Evaluation** — post-process and score (ONNX / NPZ paths)
5. **Inference** (optional) — on-target C++ FPS and realtime demos

> **Environment:** Use the Vitis AI Docker image and `conda activate vitis-ai-pytorch` for quantization and inspection.  
> **Ultralytics:** The Vitis path expects **`ultralytics==8.4.24`** (see `quantization/requirements.txt`). Root training uses the same version via `requirements.txt`. Patch Ultralytics separately in each Vitis step (see below).

## Directory layout

```text
vitis-ai/
  inspection/            # NNDCT Inspector before quantization (optional)
  quantization/          # vai_q_yolo.py, calibration data, custom_layers patch
  compilation/           # vai_c_xir inputs (model/) + Architectures/
  evaluation/
    Quantized_Model/     # ONNX eval (detect or OBB via --task)
    Compiled_Model/      # FPGA NPZ inference + eval
  inference/
    YOLOv26/             # Realtime client/server + video (YOLOv26)
    YOLOv11/             # C++ FPS + video (YOLOv11 / YOLOv11-OBB)
```

## Quick pipeline

### 0. Inspect (optional)

```bash
cd vitis-ai/quantization && python patch_ultralytics.py
cd ../inspection
python inspection.py --model_path best.pt --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B4096
```

See [inspection/README.md](inspection/README.md).

### 1. Quantize

```bash
cd vitis-ai/quantization
pip install -r requirements.txt
python patch_ultralytics.py
python prepare_calibration_data.py
python vai_q_yolo.py --model_path best.pt --batch_size 16 --img_height 416 --img_width 416 \
  --target DPUCZDX8G_ISA1_B4096 --quant_mode calib
python vai_q_yolo.py --model_path best.pt --batch_size 1 --img_height 416 --img_width 416 \
  --target DPUCZDX8G_ISA1_B4096 --quant_mode test --deploy
```

Outputs (under `quantization/quantize_result/`):

- `DetectionModel_int.xmodel` or `OBBModel_int.xmodel`
- `{checkpoint_stem}_config_no_srd_reg_nc_dfl.pkl` (e.g. `best_config_no_srd_reg_nc_dfl.pkl`)

See [quantization/README.md](quantization/README.md).

### 2. Compile

```bash
cd vitis-ai/compilation
# Copy .xmodel files into model/ first (see model/README.md)
# Uncomment the target DPU block in run_compile.sh, then run the matching line
```

See [compilation/README.md](compilation/README.md).

### 3. Evaluate

- **ONNX (quantized):** [evaluation/README.md](evaluation/README.md) — `Quantized_Model/eval_onnx.py`
- **FPGA NPZ (compiled):** `Compiled_Model/fpga_inference.py` → `eval_predictions_npz.py`

Pass the config pickle from quantization to both evaluation paths.

See [evaluation/README.md](evaluation/README.md).

### 4. Run on device (optional)

See [inference/README.md](inference/README.md).

## Artifact handoff

| From | To | Files |
|------|-----|--------|
| `quantization/quantize_result/` | `compilation/model/` | `DetectionModel_int.xmodel`, `OBBModel_int.xmodel` |
| `quantization/quantize_result/` | `evaluation/*/` | `*_config_no_srd_reg_nc_dfl.pkl` |
| `compilation/zynq_output/` | `inference/**/model/` | Compiled `.xmodel` for on-target apps |

Rename or copy `quantize_result` after each run if you keep multiple DPU sizes (see `quantization/run_compression.sh`).
