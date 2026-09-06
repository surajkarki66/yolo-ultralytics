# Realtime video inference — model files

Place compiled `.xmodel` and the matching config pickle in this folder.

| File | Notes |
|------|--------|
| `*.xmodel` | From `compilation/zynq_output/` |
| `*_config*.pkl` | From `quantization/quantize_result/` (`{stem}_config_no_srd_reg_nc_dfl.pkl`) |

Example usage (from `Video_Inference/`):

```bash
python realtime_inference_fpga_yolov8_detect.py \
  model/yolov8n.xmodel \
  model/best_config_no_srd_reg_nc_dfl.pkl \
  input.mp4 ./out --preprocess resize

python realtime_inference_fpga_yolov8_obb.py \
  model/yolov8n_obb.xmodel \
  model/best_config_no_srd_reg_nc_dfl.pkl \
  input.mp4 ./out --preprocess letterbox
```

Optional: `clean_config.py` to strip DFL/Identity nodes from a pickle before deployment.

Accuracy workflow: [../../../evaluation/README.md](../../../evaluation/README.md).
