# On-target inference

Run compiled models on FPGA / Kria hardware after [compilation](../compilation/README.md).

## Layout

```text
inference/
  YOLOv26/
    realtime_inference/        # Client + FPGA server
    realtime_inference_video/  # Video / webcam demos
  YOLOv11/
    cpp_fps_testing/
      YOLOv11/                 # Detect FPS benchmark (C++)
      YOLOv11-OBB/             # OBB FPS benchmark (C++)
    realtime_inference_video/  # Video / webcam demos
```

Use **YOLOv26** paths for YOLOv26 checkpoints; **YOLOv11** paths for YOLOv11 checkpoints (same Ultralytics-based pipeline).

## Model artifacts

Copy into each app’s `model/` folder (see README in that folder):

- Compiled `.xmodel` from `compilation/zynq_output/`
- Config pickle from `quantization/quantize_result/*_config_no_srd_reg_nc_dfl.pkl`

## C++ FPS testing (YOLOv11)

```bash
cd vitis-ai/inference/YOLOv11/cpp_fps_testing/YOLOv11
# Place .xmodel + .pkl in model/, then:
./build_app.sh
./fps_performance.sh
```

OBB: same under `YOLOv11-OBB/`.

## Realtime / video

**YOLOv26 detect:**

```bash
cd vitis-ai/inference/YOLOv26/realtime_inference_video
python realtime_inference_fpga_yolov26_detect.py ...
```

**YOLOv11 detect / OBB:**

```bash
cd vitis-ai/inference/YOLOv11/realtime_inference_video
python realtime_inference_fpga_yolov11_detect.py ...
python realtime_inference_fpga_yolov11_obb.py ...
```

**YOLOv26 client/server:**

```bash
cd vitis-ai/inference/YOLOv26/realtime_inference
# fpga_server.py on board, client_app.py on host
```

For accuracy validation before on-target deployment, use [../evaluation/README.md](../evaluation/README.md).
