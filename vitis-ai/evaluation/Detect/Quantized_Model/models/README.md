# Optional model artifacts (Detect / Quantized)

This folder is for **local copies** of models used with `post_processing_onnx.py` in the parent directory. Scripts accept paths via CLI; nothing here is required if you pass absolute paths.

Typical files:

| File | Source |
|------|--------|
| `*.onnx` | Quantized ONNX from Vitis export |
| `*_config_no_srd_reg_nc_dfl.pkl` | `quantization/quantize_result/{stem}_config_...pkl` |

Example pickle name for checkpoint `best.pt`: `best_config_no_srd_reg_nc_dfl.pkl`

Example run (from `Quantized_Model/`):

```bash
python post_processing_onnx.py \
  --onnx models/yolov8n.onnx \
  --pkl models/best_config_no_srd_reg_nc_dfl.pkl \
  --images-dir /path/to/images \
  --iou 0.45 --img-height 416 --img-width 416
```
