# Vitis AI Quantization

Scripts in this folder prepare and quantize YOLO models for AMD DPU targets. The same flow applies to **YOLOv26** and **YOLOv11** (and their OBB variants) for the Vitis-compatible export path in this repository.

## Contents

- `patch_ultralytics.py` - patches installed Ultralytics in the Vitis environment
- `prepare_calibration_data.py` - creates calibration input set
- `vai_q_yolo.py` - runs calibration/test quantization flow
- `run_compression.sh` - helper shell wrapper
- `data/` - calibration data location

## Recommended Flow

1. Patch Ultralytics in the Vitis container:

```bash
python patch_ultralytics.py
```

2. Run quantization:

```bash
python vai_q_yolo.py --model_name <converted_model.pt> --batch_size 16 --target DPUCZDX8G_ISA1_B4096 --quant_mode calib
python vai_q_yolo.py --model_name <converted_model.pt> --batch_size 16 --target DPUCZDX8G_ISA1_B4096 --quant_mode test
```

3. Send generated `.xmodel` to `../compilation/`.

Use Vitis AI Docker + `vitis-ai-pytorch` conda environment for compatibility.