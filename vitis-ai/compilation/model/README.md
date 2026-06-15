# Compilation model inputs

Place quantized `.xmodel` files here before running `vai_c_xir` (see `../run_compile.sh`).

| File | Model type |
|------|------------|
| `DetectionModel_int.xmodel` | YOLOv26 / YOLOv11 detection |
| `OBBModel_int.xmodel` | YOLOv26-OBB / YOLOv11-OBB |

Typical source: `../../quantization/quantize_result/` after `vai_q_yolo.py --quant_mode test --deploy`.

Keep Detect and OBB exports separate; compilation and evaluation use different output names (`yolov26n` vs `yolov26n_obb`).
