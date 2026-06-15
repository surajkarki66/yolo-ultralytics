# C++ FPS benchmark — YOLOv11 Detect model files

Place compiled artifacts here for `build_app.sh` and `fps_performance.sh`.

| File | Typical source |
|------|----------------|
| `*.xmodel` | `compilation/zynq_output/yolov11n/` (or your compile `-n` name) |
| `*_config_no_srd_reg_nc_dfl.pkl` | `quantization/quantize_result/best_config_no_srd_reg_nc_dfl.pkl` |

Example names used in scripts (pass stem to `fps_performance.sh`):

- `yolov11n.xmodel`
- `yolov11n_config_no_srd_reg_nc_dfl.pkl` (rename pickle if needed)

For NPZ-based accuracy checks, see [../../../../evaluation/README.md](../../../../evaluation/README.md).
