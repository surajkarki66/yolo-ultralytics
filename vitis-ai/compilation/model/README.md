# Compilation model inputs

Place quantized `.xmodel` files here before running `vai_c_xir` (see `../run_compile.sh`).

| File | Model type |
|------|------------|
| `DetectionModel_int.xmodel` | YOLOv8 detection |
| `OBBModel_int.xmodel` | YOLOv8-OBB |

Typical source: `../../quantization/quantize_result/` after `vai_q_yolo.py --quant_mode test --deploy`.

Keep Detect and OBB exports separate; compilation and evaluation use different output names (`yolov8n` vs `yolov8n_obb`).
