# On-target inference

Run compiled models on FPGA / Kria hardware after [compilation](../compilation/README.md).

## Layout

```text
inference/
  cpp_fps_testing/
    YOLOv8/          # Detect FPS benchmark (C++)
    YOLOv8-OBB/      # OBB FPS benchmark (C++)
  realtime_inference/
    Client/          # Remote client app
    FPGA_Server/     # Server on board
    Video_Inference/ # Video / realtime demos
```

## Model artifacts

Copy into each app’s `model/` folder (see README in that folder):

- Compiled `.xmodel` from `compilation/zynq_output/`
- Config pickle from `quantization/quantize_result/*_config_no_srd_reg_nc_dfl.pkl`

## C++ FPS testing

```bash
cd vitis-ai/inference/cpp_fps_testing/YOLOv8
# Place .xmodel + .pkl in model/, then:
./build_app.sh
./fps_performance.sh
```

OBB: same under `YOLOv8-OBB/`.

## Realtime / video

```bash
cd vitis-ai/inference/realtime_inference/Video_Inference
# See model/README.md for expected filenames
python realtime_inference_fpga_yolov8_detect.py ...
python realtime_inference_fpga_yolov8_obb.py ...
```

For accuracy validation before on-target deployment, use [../evaluation/README.md](../evaluation/README.md).
