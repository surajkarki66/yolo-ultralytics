# Vitis AI deployment pipeline

End-to-end flow for deploying YOLOv8 **Detect** and YOLOv8 **OBB** on AMD DPU (Zynq/Kria) using Vitis AI:

1. **Inspection** (optional) — validate DPU compatibility before quantization
2. **Quantization** — calibrate and export `.xmodel` (+ config pickle)
3. **Compilation** — build device-ready artifacts with `vai_c_xir`
4. **Evaluation** — post-process and score (ONNX / NPZ paths)
5. **Inference** (optional) — C++ FPS benches and realtime demos on target

> **Environment:** Use the Vitis AI Docker image and `conda activate vitis-ai-pytorch` for quantization.  
> **Ultralytics:** The quantization path expects **`ultralytics==8.1.47`** (see `quantization/requirements.txt`). The main repo training stack may use a newer Ultralytics version via root `requirements.txt`.

## Directory layout

```text
vitis-ai/
  inspection/            # NNDCT Inspector before quantization (optional)
  quantization/          # vai_q_yolo.py, calibration data, custom_layers patch
  compilation/           # vai_c_xir inputs (model/) + DPU arch JSON
  evaluation/
    Detect/
      Quantized_Model/   # ONNX → boxes → COCO metrics
      Compiled_Model/    # FPGA NPZ → boxes → COCO metrics
    OBB/
      Quantized_Model/   # ONNX → OBB metrics (no pycocotools)
      Compiled_Model/    # FPGA NPZ → OBB metrics / visualization
  inference/
    cpp_fps_testing/     # On-board FPS (C++)
    realtime_inference/  # Client / server / video demos
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

- `DetectionModel_int.xmodel` or deploy export name from Vitis
- `{checkpoint_stem}_config_no_srd_reg_nc_dfl.pkl` (e.g. `best_config_no_srd_reg_nc_dfl.pkl`)

See [quantization/README.md](quantization/README.md).

### 2. Compile

```bash
cd vitis-ai/compilation
# Copy .xmodel files into model/ first (see model/README.md)
# Uncomment the target DPU block in run_compile.sh, then:
bash run_compile.sh
```

See [compilation/README.md](compilation/README.md).

### 3. Evaluate

- **Detect:** [evaluation/README.md](evaluation/README.md) — `Detect/Quantized_Model` (ONNX) or `Detect/Compiled_Model` (NPZ from FPGA)
- **OBB:** same README — `OBB/Quantized_Model` or `OBB/Compiled_Model`

### 4. Run on device (optional)

See [inference/README.md](inference/README.md).

## Artifact handoff

| From | To | Files |
|------|-----|--------|
| `quantization/quantize_result/` | `compilation/model/` | `DetectionModel_int.xmodel`, `OBBModel_int.xmodel` |
| `quantization/quantize_result/` | `evaluation/*/` | `*_config_no_srd_reg_nc_dfl.pkl` |
| `compilation/zynq_output/` | `inference/**/model/` | Compiled `.xmodel` for on-target apps |

Rename or copy `quantize_result` after each run if you keep multiple DPU sizes (see `run_compression.sh`).
