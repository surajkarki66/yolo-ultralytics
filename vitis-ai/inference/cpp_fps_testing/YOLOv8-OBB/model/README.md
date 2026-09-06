# C++ FPS benchmark — OBB model files

Place compiled OBB artifacts here for `YOLOv8-OBB/build_app.sh` and `fps_performance.sh`.

| File | Typical source |
|------|----------------|
| `*.xmodel` | `compilation/zynq_output/yolov8n_obb/` |
| `*_config_no_srd_reg_nc_dfl.pkl` | `quantization/quantize_result/` (OBB training checkpoint stem) |

Rename files to match what your build scripts expect, or update the shell scripts accordingly.

Evaluation: [../../../evaluation/README.md](../../../evaluation/README.md) → `OBB/Compiled_Model/`.
