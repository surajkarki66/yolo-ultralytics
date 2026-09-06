# C++ FPS benchmark — Detect model files

Place compiled artifacts here for `YOLOv8/build_app.sh` and `fps_performance.sh`.

| File | Typical source |
|------|----------------|
| `*.xmodel` | `compilation/zynq_output/yolov8n/` (or your compile `-n` name) |
| `*_config_no_srd_reg_nc_dfl.pkl` | `quantization/quantize_result/best_config_no_srd_reg_nc_dfl.pkl` |

Example names used in scripts:

- `yolov8n.xmodel`
- `yolov8n_config_no_srd_reg_nc_dfl.pkl` (rename pickle if needed)

For NPZ-based accuracy checks, see [../../../evaluation/README.md](../../../evaluation/README.md).
