# Vitis AI evaluation

Post-processing and metrics for **YOLOv26** / **YOLOv11** **Detect** and **OBB** models.

Two deployment paths:

| Path | Folder | When to use |
|------|--------|-------------|
| **Quantized (ONNX)** | `Quantized_Model/` | PC-side ONNX from quantization deploy |
| **Compiled (FPGA)** | `Compiled_Model/` | DPU after `vai_c_xir` compilation |

Both paths need the **config pickle** from quantization:

`quantize_result/{checkpoint_stem}_config_no_srd_reg_nc_dfl.pkl`  
(e.g. `best_config_no_srd_reg_nc_dfl.pkl` for `best.pt`)

Use `--task detect` or `--task obb` to select the head type.

## Quantized (ONNX)

```bash
cd vitis-ai/evaluation/Quantized_Model
python eval_onnx.py \
  --task detect \
  --model path/to/model.onnx \
  --data path/to/data.yaml \
  --quant-meta path/to/best_config_no_srd_reg_nc_dfl.pkl \
  --imgsz 416
```

For OBB, set `--task obb`. Writes `metrics.json` and optional visualizations under `--save-dir`.

## Compiled (FPGA NPZ)

**Step 1 — run inference on device or Vitis runtime:**

```bash
cd vitis-ai/evaluation/Compiled_Model
python fpga_inference.py model/yolov26n.xmodel ./images predictions.npz 416
# OBB: use fpga_inference_obb.py instead
```

**Step 2 — evaluate NPZ predictions:**

```bash
python eval_predictions_npz.py \
  --predictions-npz predictions.npz \
  --task detect \
  --data path/to/data.yaml \
  --quant-meta path/to/best_config_no_srd_reg_nc_dfl.pkl \
  --imgsz 416
```

## Layout

| File | Role |
|------|------|
| `Quantized_Model/eval_onnx.py` | ONNX Runtime eval for detect or OBB |
| `Quantized_Model/utils.py` | Metrics, NMS, YAML helpers |
| `Compiled_Model/fpga_inference.py` | Detect model → NPZ on FPGA |
| `Compiled_Model/fpga_inference_obb.py` | OBB model → NPZ on FPGA |
| `Compiled_Model/eval_predictions_npz.py` | Decode NPZ + mAP for detect or OBB |
| `Compiled_Model/utils.py` | Shared metrics / NMS helpers |

## Notes

- Match `--imgsz` to training, quantization, and compilation input size (e.g. 416).
- `--quant-meta` loads head metadata (`no`, stride, `reg_max`, `nc`, DFL) exported during quantization calib.
- Detect uses 3 outputs; OBB uses 6 outputs (3 box + 3 angle heads).
